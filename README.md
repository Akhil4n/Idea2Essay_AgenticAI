# Agentic AI Video Generator

## Overview
This project demonstrates a **real-time AI video generation workflow** built with **four distinct AI agents** that collaborate sequentially to create 10-second AI videos from any topic. Using OpenAI's GPT-4o-mini API for planning and Replicate's AI video models for generation, each agent performs a specific task: planning, scene design, visual description, and video creation. The system includes a **graphical web interface** built with Flask and JavaScript for users to enter topics and watch each agent's action live via streaming updates.

## Features
- **Four distinct AI agents** forming a sequential video workflow (planner → scene planner → visual writer → video generator)
- User submits topics via web GUI; results display **step-by-step outputs dynamically** with live video playback
- Backend built on Flask serving API endpoints, SSE streaming, and video serving
- **OpenAI GPT-4o-mini** for AI agent tasks + **Replicate** for AI video generation
- **Topic-based video filenames** saved to `videos/` folder (`solar_panels_1732849201.mp4`)
- **Auto-created `videos/` folder** - no manual setup required
- **Pure visual videos** - no text/subtitles for professional results
- Easy API key management via `.env` file and python-dotenv

## Setup
1. Clone or download the repository
2. Create a `.env` file in the root directory with your API keys:
OPENAI_API_KEY=sk-proj-your-key-here
REPLICATE_API_TOKEN=r8-your-token-here

3. Install Python dependencies:
pip install -r requirements.txt


## Running

Start the Flask server:
python app.py


Open your browser and go to:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

Enter a topic like **"Lamborghini** and click **Generate 10s AI Video** to watch the 4 agents create your video live!

## Project Structure

| Path                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `app.py`              | Flask backend + 4 AI agents + video serving  |
| `index.html`          | Streaming web UI (HTML + vanilla JS)         |
| `.env`                | Stores OpenAI + Replicate API keys           |
| `requirements.txt`    | Python packages required                     |
| `videos/`             | Auto-created folder for generated videos     |

## Output Formatting

- **Agents A & B** produce markdown-style bullet lists, rendered in `<pre>` blocks
- **Agent C** outputs visual descriptions with line breaks preserved
- **Agent D** shows final video prompt + **plays generated video** + displays filename
- **Videos auto-save** as `videos/topic_timestamp.mp4` in videos folder in your working directory
