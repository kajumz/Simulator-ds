import os
import re
from uuid import uuid4

import streamlit as st

# TODO: Uncomment these lines when you have implemented the functions
from src.download import convert_mp4_to_mp3, download_audio, video_title
from src.transcribe import transcribe
from src.summarize import summarize_text


def main():
    st.title("Видео Суммаризатор")
    os.environ['OPENAI_API_KEY'] = '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'
    # Paste url to youtube video
    youtube_url = st.text_input("Вставьте ссылку на видеоролик в youtube:")

    # Regex check youtube url
    if re.match(r"^https://www.youtube.com/watch\?v=[a-zA-Z0-9_-]*$", youtube_url):
        # Display video
        st.video(youtube_url)

        transcribe_button = st.empty()
        title_placeholder = st.empty()
        progress_placeholder = st.empty()

        # Button to download audio from youtube video
        if transcribe_button.button("Суммаризировать видео"):
            # Download audio
            try:
                transcribe_button.empty()

                # TODO: Change template
                title_placeholder.title(str(video_title(youtube_url=youtube_url)))

                progress_placeholder.text("Скачиваю видео...")

                # TODO: Create a runtimes folder and runtime id
                runtime = 'D:/pythonProject4/junior/video summary/video-summary/runtimes'
                os.makedirs(runtime, exist_ok=True)
                runtime_id = str(uuid4())
                mp4_path = f"runtimes/{runtime_id}.mp4"
                mp3_path = f"runtimes/{runtime_id}.mp3"
                # Download audio to runtimes/ folder
                download_audio(youtube_url=youtube_url, download_path=mp4_path)

                # Convert mp4 to mp3
                convert_mp4_to_mp3(mp4_path, mp3_path)
            except Exception as e:
                print(e)
                st.error("Пожалуйста, предоставьте корректную ссылку на видео!")
                transcribe_button.empty()
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            # Transcribe
            try:
                progress_placeholder.text("Распознавание аудио...")

                # TODO: Transcribe audio
                video_text = transcribe(mp3_path, "base")
                print(video_text)
            except Exception as e:
                print(e)
                st.error("Ошибка распознавания. Пожалуйста, попробуйте еще раз!")
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            # Summarize
            try:
                assert os.environ["OPENAI_API_KEY"], "OPENAI_API_KEY not found!"

                progress_placeholder.text("Суммаризация...")

                # TODO: Summarize text
                summary = summarize_text(video_text)

                st.text_area("Результат", summary, height=300)
            except Exception as e:
                print(e)
                st.error("Ошибка суммаризации. Пожалуйста, попробуйте еще раз!")
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            progress_placeholder.empty()


if __name__ == "__main__":
    main()
