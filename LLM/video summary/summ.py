import openai


def summary_prompt(input_text: str) -> str:
    """
    Build prompt using input text of the video.
    """
    prompt = f"""
    Summarize the following text:
    {input_text}
    """
    return prompt


def summarize_text(input_text: str) -> str:
    """
    Summarize input text of the video.

    Examples
    --------
    #>>> summary = summarize_text(video_text)
    #>>> print(summary)
    'This video explains...'
    """
    openai.api_key = ''


    # Send request to OpenAI

    # Generate response
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': summary_prompt(input_text)}],
        temperature=0.5
    )
    # Extract summary from response
    summary = response.choices[0].message['content']
    return summary

    # Return response




