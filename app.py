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

os.makedirs("videos", exist_ok=True)

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


# Helpers

def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[^\w\s-]', '', filename).strip()
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename[:50]

def get_video_filename(user_topic: str) -> str:
    clean_topic = sanitize_filename(user_topic) or "video"
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


# Agents 

def agent_a_planner(user_topic: str) -> str:
    instructions = (
        "You are Agent A, planning a 10-second AI video explainer.\n"
        "Given the user topic, create a simple visual outline.\n"
        "Output format:\n"
        "- Target duration: 10 seconds\n"
        "- Intro visuals (1 bullet)\n"
        "- Core idea visuals (1 bullet)\n"
        "- Outro visuals (1 bullet)\n"
        "Focus ONLY on what is shown, not what is said.\n"
        "Do NOT include any on-screen text, labels, titles, or subtitles."
    )
    return call_openai_agent(instructions, user_topic)


def agent_b_scenes_and_visuals(outline_text: str) -> str:
    instructions = (
        "You are Agent B, designing 3 detailed cinematic scenes for a 10-second AI video.\n"
        "Use these time ranges exactly: 0-3s, 3-6s, 6-10s.\n"
        "For each scene, include:\n"
        "- Scene number and short title\n"
        "- Time range\n"
        "- Rich visual description including motion, colors, style, and camera movement.\n"
        "Do NOT include any on-screen text, subtitles, labels, UI, or written words of any kind.\n"
        "Focus only on what the camera sees and how it moves."
    )
    return call_openai_agent(instructions, outline_text)


def agent_c_final_prompt(scene_plan_text: str, user_topic: str) -> str:
    instructions = (
        "You are Agent C, an expert text-to-video prompt engineer.\n"
        "Your job is to transform a multi-scene plan into ONE concise prompt "
        "for generating a 10-second AI video.\n"
        "Requirements:\n"
        "- Clearly state the overall subject based on the user topic.\n"
        "- Describe how the visuals evolve over the full 10 seconds.\n"
        "- Include motion and camera dynamics.\n"
        "- DO NOT include any on-screen text, subtitles, captions, UI, or written words.\n"
        "- Output a single descriptive paragraph, no bullet points."
    )
    combined = f"User topic: {user_topic}\n\nScene plan:\n{scene_plan_text}"
    return call_openai_agent(instructions, combined)


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

        filename = get_video_filename(user_topic)
        print(f"Downloading video to: {filename}")
        video_content = requests.get(str(output), timeout=60).content
        with open(filename, "wb") as f:
            f.write(video_content)
        print(f"Saved video file: {filename}")

        return {
            "status": "completed",
            "filename": filename,
            "source_url": str(output),
        }
    except Exception as e:
        print(f"Video error: {str(e)}")
        return {"status": "error", "error": str(e), "filename": None}


def agent_d_video_generator(final_prompt: str, user_topic: str) -> dict:
    video_info = generate_video_from_prompt(final_prompt, user_topic)

    video_url = None
    if video_info["status"] == "completed" and video_info["filename"]:
        video_url = f"/videos/{os.path.basename(video_info['filename'])}"

    return {
        "final_prompt": final_prompt,
        "video_info": video_info,
        "video_url": video_url,
    }


# Routes

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
            print("Running Agent A (outline)...")
            a_out = agent_a_planner(prompt)
            yield f"data: {json.dumps({'agent': 'A', 'content': a_out})}\n\n"

            print("Running Agent B (scenes + visuals)...")
            b_out = agent_b_scenes_and_visuals(a_out)
            yield f"data: {json.dumps({'agent': 'B', 'content': b_out})}\n\n"

            print("Running Agent C (final prompt)...")
            c_out = agent_c_final_prompt(b_out, prompt)
            yield f"data: {json.dumps({'agent': 'C', 'content': c_out})}\n\n"

            print("Running Agent D (video generation)...")
            d_obj = agent_d_video_generator(c_out, prompt)
            event_data = {
                "agent": "D",
                "content": d_obj["final_prompt"],
                "video_url": d_obj["video_url"],
                "video_status": d_obj["video_info"]["status"],
                "filename": d_obj["video_info"].get("filename", "N/A"),
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            yield f"data: {json.dumps({'agent': 'DONE', 'content': 'Complete! Video saved.'})}\n\n"

        except Exception as e:
            error_event = {"agent": "ERROR", "content": f"Error: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return Response(event_stream(user_prompt), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
