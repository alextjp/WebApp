# gpt_prompts.py

PLANNER_PROMPTS = {
    "header": """
    - Input: Propose enhancements to make the header more engaging.
    - Output: Increase the header size, use a bold font, and add a call-to-action button.
    """,
    
    "footer": """
    - Input: Suggest improvements for the footer layout.
    - Output: Include links to important pages, add social media icons, and ensure clear contact information.
    """,
    
    "main_content": """
    - Input: Improve the main content readability and engagement.
    - Output: Use shorter paragraphs, include bullet points, and add relevant images or infographics.
    """,
    
    "navigation": """
    - Input: Enhance the navigation bar for better user experience.
    - Output: Use a fixed navigation bar, highlight the current section, and ensure responsive design.
    """,
    
    "SEO": """
    - Input: Suggest SEO improvements for the webpage.
    - Output: Add a meta description tag, optimize image alt texts, and ensure header tags are used effectively.
    """,
    
    # More examples as needed
}

OUTPUT_PROMPTS = {
    "header": """
    - Task: Enhance the webpage header
    - LLM Output: Increase the header size...
    - Action: Apply the suggested changes to the header section of the HTML.
    """,
    
    "footer": """
    - Task: Improve the footer layout
    - LLM Output: Include links to important pages...
    - Action: Update the footer section with the suggested links and information.
    """,
    
    "main_content": """
    - Task: Improve readability and engagement of the main content
    - LLM Output: Use shorter paragraphs...
    - Action: Edit the main content area by applying the suggestions from the LLM.
    """,
    
    "navigation": """
    - Task: Enhance the navigation bar
    - LLM Output: Use a fixed navigation bar...
    - Action: Implement the navigation bar enhancements in the HTML.
    """,
    
    "SEO": """
    - Task: Optimize the webpage for SEO
    - LLM Output: Add a meta description tag...
    - Action: Update the HTML to include the recommended SEO improvements.
    """,
    
    # More examples as needed
}
