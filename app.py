from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai_agent(instructions: str, user_input: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=instructions,
        input=user_input
    )
    return response.output_text

def agent_a_planner(user_topic: str) -> str:
    instructions = (
        "You are Agent A, a video content planner. "
        "Given a topic, create a concise outline for a short, informational video. "
        "Output:\n"
        "- Target video length (between 25 and 50 seconds).\n"
        "- 3–5 bullet points: Intro, 2–3 key teaching points, Outro.\n"
        "Use clear markdown bullet points."
    )
    return call_openai_agent(instructions, user_topic)

def agent_b_scene_planner(outline_text: str) -> str:
    instructions = (
        "You are Agent B, a video scene planner. "
        "Turn the outline into a numbered list of scenes for a short video. "
        "For each scene, include:\n"
        "- Scene number and short title.\n"
        "- Time range in seconds (e.g., 0–10s, 10–25s).\n"
        "- Visual description (what the viewer sees).\n"
        "- On-screen text or key words if any.\n"
        "Format as markdown with numbered scenes and sub-bullets."
    )
    return call_openai_agent(instructions, outline_text)

def agent_c_script_writer(scene_plan_text: str) -> str:
    instructions = (
        "You are Agent C, a narration script writer. "
        "Based on the scene plan, write the spoken narration for a short, "
        "informational video. Group lines by scene.\n"
        "For each scene:\n"
        "- Put a scene heading, e.g., 'Scene 1 – Intro'.\n"
        "- Under it, write 2–4 short sentences of narration.\n"
        "Keep the language simple and clear."
    )
    return call_openai_agent(instructions, scene_plan_text)

def agent_d_editor(narration_text: str) -> str:
    instructions = (
        "You are Agent D, a video brief editor. "
        "Create a final video production brief from the scene plan and narration that can be effectively used as input for a video generation AI. "
        "Summarize:\n"
        "- Video title.\n"
        "- Target audience and tone.\n"
        "- Approximate total duration.\n"
        "- For each scene: time range, what to show, and which narration lines to use.\n"
        "Format the result with clear section headings and short paragraphs."
    )
    return call_openai_agent(instructions, narration_text)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_workflow", methods=["POST"])
def run_workflow():
    data = request.get_json()
    user_prompt = data.get("prompt", "")

    a_out = agent_a_planner(user_prompt)
    b_out = agent_b_scene_planner(a_out)
    c_out = agent_c_script_writer(b_out)
    d_out = agent_d_editor(c_out)

    return jsonify({
        "agent_outputs": {
            "A": a_out,
            "B": b_out,
            "C": c_out,
            "D": d_out
        },
        "final": d_out
    })

if __name__ == "__main__":
    app.run(debug=True)
