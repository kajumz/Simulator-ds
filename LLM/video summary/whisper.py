import torch.cuda
import whisper


def transcribe(file_path: str, model_name="base") -> str:
    """
    Transcribe input audio file.

    Examples
    --------
    #>>> text = transcribe(".../audio.mp3")
    #>>> print(text)
    'This text explains...'
    """
    # YOUR CODE HERE
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

        # Load the transcription model
    model = whisper.load_model(model_name, device=device)

    # Set the device for the model


    # Transcribe the audio
    result = model.transcribe(file_path)

    # Return the transcription as text
    return result["text"]

