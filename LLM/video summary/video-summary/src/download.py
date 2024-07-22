from pytube import YouTube
from moviepy.editor import *

def video_title(youtube_url: str) -> str:
    """
    Retrieve the title of a YouTube video.

    Examples
    --------
    #>>> title = video_title("https://www.youtube.com/watch?v=SampleVideoID")
    #>>> print(title)
    'Sample Video Title'
    """
    # YOUR CODE HERE
    yt = YouTube(youtube_url)
    title = yt.title
    return title



def download_audio(youtube_url: str, download_path: str) -> None:
    """
    Download the audio from a YouTube video.

    Examples
    --------
    #>>> download_audio("https://www.youtube.com/watch?v=SampleVideoID", "path/to/save/audio.mp4")
    """
    # YOUR CODE HERE
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
    output_directory, filename = os.path.split(download_path)
    audio_stream.download(output_path=output_directory, filename=filename)

def convert_mp4_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert an audio file from mp4 format to mp3.

    Examples
    --------
    #>>> convert_mp4_to_mp3("path/to/audio.mp4", "path/to/audio.mp3")
    """
    # YOUR CODE HERE
    audio = AudioFileClip(input_path)
    audio.write_audiofile(output_path, codec='mp3')

