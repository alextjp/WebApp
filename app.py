# app.py

import os
import openai
import logging
import traceback
import asyncio
import difflib
from diff_match_patch import diff_match_patch

from bs4 import BeautifulSoup
from flask import Flask, request, render_template, jsonify

from llm_compiler import compiler
from llm_compiler.agent import LLM, WebDevAgent, WebPageProcessor
from llm_compiler.tools import WebContentTool, DalleTool
from llm_compiler.chat_agent import WebDevChatAgent
from llm_compiler.compiler import WebDevLLMCompiler
from llm_compiler.llm_chain import LLMChain
from llm_compiler.config import openai_api_key, model_name

# Set up Flask
app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLM and tools
llm = LLM(api_key=openai_api_key, model_name=model_name)
web_content_tool = WebContentTool()
dalle_tool = DalleTool()
tools = [web_content_tool, dalle_tool]

# Initialize the WebPageProcessor with the LLM
webpage_processor = WebPageProcessor(llm=llm)

# Initialize the Web Development Agent with LLM, tools, and webpage_processor
web_dev_agent = WebDevAgent(llm=llm, tools=tools, webpage_processor=webpage_processor)

# Initialize the LLM Compiler and Web Development Agent
openai_api_key = os.getenv('openai_api_key')
if not openai_api_key:
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Configure global definitions, model name and tools
model_name = "gpt-3.5-turbo-1106"
dalle_tool = DalleTool()
app_state = {'original_html': ''}
global_original_html = None
global_modified_html = None

# Initialize various instances
llm_instance = LLM(api_key=openai_api_key, model_name=model_name)

llm_chain = LLMChain(
    llm=llm_instance,
    prompt="Your prompt here",
    model_name=model_name,
    api_key=openai_api_key)
llm_compiler = WebDevLLMCompiler(
    tools=tools,
    llm=llm_instance,  # The llm instance using openai
    prompt="Your prompt here",
    max_tokens=500)

chat_agent = WebDevChatAgent(llm=llm_instance, tools=tools)

def make_diff(old_html, new_html):
    diff = difflib.HtmlDiff().make_file(old_html.splitlines(), new_html.splitlines())
    return diff

def track_changes(old_html, new_html):
    dmp = difflib.ndiff(old_html.splitlines(), new_html.splitlines())
    return '\n'.join(dmp)

async def async_run(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

def make_diff(old_html, new_html):
    diff = difflib.HtmlDiff().make_file(old_html.splitlines(), new_html.splitlines())
    return diff

def track_changes(old_html, new_html):
    dmp = diff_match_patch()
    diff = dmp.diff_main(old_html, new_html)
    return diff

def run_asyncio(async_method):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_method)

@app.route('/process_input', methods=['POST'])
async def process_input():
    data = request.get_json()
    user_input = data.get('user_input', '')

    try:
        result = async_run(llm_compiler.generate_content, user_input)
        original_html, modified_html = web_dev_agent.process_input(user_input)
        changes = track_changes(original_html, modified_html)
        return jsonify({'result': result, 'changes': changes}), 200
    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Get the full traceback
    error_info = traceback.format_exc()
    logger.error("Global Error: %s\n%s", e, error_info)
    # Render error.html with error details
    return render_template("error.html", error=str(e), error_detail=error_info), 500


@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 Error: Page not found: {request.url}")
    return "404 Error: Page not found", 404


@app.errorhandler(403)
def forbidden(e):
    logger.error(f"403 Forbidden: Access denied for {request.url}")
    return "403 Forbidden: Access denied", 403


@app.route('/', methods=['GET', 'POST'])
def index():
    llm_instance = LLM(api_key=openai_api_key, model_name="gpt-3.5-turbo-1106")
    llm_compiler = WebDevLLMCompiler(tools, llm_instance, "Modify Page", 500)

    try:
        if request.method == 'POST':
            user_input = request.form.get('prompt')
            result = run_asyncio(llm_compiler.generate_content(user_input))  # Correction here
            # Process the input through the LLM Compiler and Web Development Agent
            response = llm_compiler.process_input(user_input)
            modified_content = web_dev_agent.modify_content(response)
            return render_template('index.html', original_content=app_state['original_html'], modified_content=modified_content)
        else:
            return render_template('index.html', original_content=app_state['original_html'])
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template("error.html", error=str(e)), 500

def extract_visible_text(html_content):
    """Extract visible plain text from HTML, excluding footer copyright text."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text and filter out empty lines and spaces
    text = '\n'.join(filter(None, (line.strip() for line in soup.get_text().splitlines())))

    return text


@app.route('/original_page')
def original_page():
    return open('templates/p_clone.html', 'r').read()


@app.route('/modified_page')
def modified_page():
    global global_modified_html
    if global_modified_html:
        return global_modified_html
    return "No modifications made yet."


@app.route('/ssr_modified')
def ssr_modified_page():
    global global_modified_html
    if global_modified_html:
        return render_template('ssr_modified_template.html', modified_html=global_modified_html)
    return render_template('ssr_modified_template.html', modified_html="No modifications made yet.")


@app.route('/mockup')
def mockup():
    return render_template('mockup.html')


@app.route('/comparison')
def comparison():
    global global_original_html
    global global_modified_html
    if global_original_html and global_modified_html:
        diffs = make_diff(global_original_html, global_modified_html)
        return diffs
    return "No changes are available yet."


@app.route('/parse', methods=['POST'])
def parse_html():
    html_content = request.form.get('html_content')
    visible_text = extract_visible_text(html_content)

    return jsonify({'visible_text': visible_text})


@app.route('/p_clone')
def p_clone():
    try:
        return open('templates/p_clone.html', 'r').read()
    except Exception as e:
        logger.error("Error serving /p_clone: %s", e)
        return "Error loading original page."


@app.route('/api/modify', methods=['POST'])
def api_modify_content():
    data = request.get_json()
    query = data.get('query', '')

    try:
        modified_text = async_run(web_dev_agent.modify_content, query)
        return jsonify({'modified_text': modified_text})
    except Exception as e:
        logger.error(f"Error modifying content: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/get_modified_content', methods=['POST'])
def get_modified_content():
    try:
        data = request.get_json()  # POST request
        user_input = data.get('user_input')
        result = run_asyncio(compiler.generate_content(user_input))  # Process it with LLM Compiler
        return jsonify({'result': result})
    except Exception as e:
        logger.error("Error modifying content: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize global_original_html at app start
    with open('templates/p_clone.html', 'r') as file:
        global_original_html = file.read()

    app.run(host='0.0.0.0', port=8000, debug=True)
    global_modified_html = None

