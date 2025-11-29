from flask import Flask, request, render_template, Response, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv
import replicate
import requests
import os
import json
import time
import re

load_dotenv()

# Create videos folder
os.makedirs("videos", exist_ok=True)

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

def sanitize_filename(filename: str) -> str:
    """Make filename safe: remove special chars, limit length"""
    filename = re.sub(r'[^\w\s-]', '', filename).strip()
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename[:50]

def get_video_filename(user_topic: str) -> str:
    """Generate topic-based filename: topic_timestamp.mp4"""
    clean_topic = sanitize_filename(user_topic)
    timestamp = int(time.time())
    return f"videos/{clean_topic}_{timestamp}.mp4"

def call_openai_agent(instructions: str, user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

def agent_a_planner(user_topic: str) -> str:
    instructions = (
        "You are Agent A, planning a 10-second AI video explainer. "
        "Create a simple visual outline - NO on-screen text needed.\n"
        "Output:\n"
        "- Target: 10 seconds\n"
        "- 3 bullet points: Intro visuals, 1 key visual demo, Outro visuals\n"
        "Focus on motion, camera moves, and visuals only."
    )
    return call_openai_agent(instructions, user_topic)

def agent_b_scene_planner(outline_text: str) -> str:
    instructions = (
        "You are Agent B, planning 3 scenes for a 10-second AI video. "
        "Time ranges: 0-3s, 3-6s, 6-10s.\n"
        "For each scene:\n"
        "- Scene number and title\n"
        "- Time range\n"
        "- Visual description with camera motion ONLY\n"
        "NO on-screen text, subtitles, or words. Pure visuals."
    )
    return call_openai_agent(instructions, outline_text)

def agent_c_script_writer(scene_plan_text: str) -> str:
    instructions = (
        "You are Agent C, describing key visuals for a 10-second AI video. "
        "For each scene, write 1 sentence describing what viewers SEE (no narration needed).\n"
        "Focus on motion, colors, camera angles. NO text elements."
    )
    return call_openai_agent(instructions, scene_plan_text)

def generate_video_from_prompt(prompt: str, user_topic: str) -> dict:
    try:
        print(f"Generating AI video: {prompt[:100]}...")
        output = replicate.run(
            "bytedance/seedance-1-pro-fast",
            input={
                "prompt": prompt,
                "duration": 10,
                "width": 640,
                "height": 360,
                "num_inference_steps": 25
            }
        )
        
        # Save to videos/ folder with topic name
        filename = get_video_filename(user_topic)
        print(f"Downloading to: {filename}")
        video_content = requests.get(str(output), timeout=30).content
        with open(filename, 'wb') as f:
            f.write(video_content)
        print(f"Saved: {filename}")
        
        return {
            "status": "completed",
            "filename": filename,
            "source_url": str(output)
        }
    except Exception as e:
        print(f"Video error: {str(e)}")
        return {"status": "error", "error": str(e), "filename": None}

def agent_d_video_generator(scene_plan_text: str, narration_text: str, user_topic: str) -> dict:
    instructions = (
        "You are Agent D, composing an AI video text-to-video prompt. "
        "Create one 10-second prompt from the scene plan + visuals.\n"
        "IMPORTANT: NO text, subtitles, words, or on-screen graphics.\n"
        "Pure clean animation: smooth motion, camera moves, visual effects only.\n"
        "One descriptive paragraph."
    )
    
    clean_scenes = scene_plan_text.replace("text", "").replace("Text", "").replace("words", "")
    clean_scenes = clean_scenes.replace("subtitle", "").replace("title", "").replace("label", "")
    
    combined = f"Visual scenes only (ignore any text mentions):\n{clean_scenes}\n\nKey visuals:\n{narration_text}"
    wan_prompt = call_openai_agent(instructions, combined)
    
    video_info = generate_video_from_prompt(wan_prompt, user_topic)
    
    video_url = None
    if video_info["status"] == "completed" and video_info["filename"]:
        video_url = f"/videos/{os.path.basename(video_info['filename'])}"

    return {
        "prompt": wan_prompt,
        "video_info": video_info,
        "video_url": video_url,
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/videos/<path:filename>")
def get_video(filename):
    return send_from_directory("videos", filename, mimetype="video/mp4")

@app.route("/run_workflow_stream", methods=["POST"])
def run_workflow_stream():
    user_prompt = request.json.get("prompt", "")

    def event_stream(prompt: str):
        try:
            print("Running Agent A...")
            a_out = agent_a_planner(prompt)
            yield f"data: {json.dumps({'agent': 'A', 'content': a_out})}\n\n"

            print("Running Agent B...")
            b_out = agent_b_scene_planner(a_out)
            yield f"data: {json.dumps({'agent': 'B', 'content': b_out})}\n\n"

            print("Running Agent C...")
            c_out = agent_c_script_writer(b_out)
            yield f"data: {json.dumps({'agent': 'C', 'content': c_out})}\n\n"

            print("Running Agent D + Video...")
            d_obj = agent_d_video_generator(b_out, c_out, prompt)
            event_data = {
                'agent': 'D', 
                'content': d_obj['prompt'],
                'video_url': d_obj['video_url'],
                'video_status': d_obj['video_info']['status'],
                'filename': d_obj['video_info'].get('filename', 'N/A')
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            yield f"data: {json.dumps({'agent': 'DONE', 'content': 'Complete! Video saved.'})}\n\n"
            
        except Exception as e:
            error_event = {'agent': 'ERROR', 'content': f"Error: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(event_stream(user_prompt), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)
