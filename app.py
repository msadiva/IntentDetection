## create gradio ui and api endpoint for the model
import gradio as gr
from utils import predict_label
from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


main_app = FastAPI(redoc_url=None, docs_url=None)


async def verify_developer_token(api_key: str = Security(api_key_header)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
    )
    if api_key is None:
        raise credentials_exception
    elif api_key.split(" ")[-1] != "6828f134":
        raise credentials_exception

    return api_key


def predict(text):
    return predict_label(text)


## post endpoint with authentication
@main_app.post("/predict")
def predict_endpoint(text: str, api_key_header: str = Depends(verify_developer_token)):
    return predict(text)


with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
    with gr.Tab("Intent Detection"):
        gr.Markdown("## Intent Detection")
        input_text = gr.Textbox(label="Text", lines=4)
        with gr.Row():
            with gr.Column():
                predict_button = gr.Button("Predict", variant="primary")
            with gr.Column():
                clear_button = gr.Button("Clear")
        output_label = gr.Label(label="Intent")

    predict_button.click(fn=predict, inputs=input_text, outputs=output_label)
    clear_button.click(
        fn=lambda: [input_text.clear(), output_label.clear()],
        inputs=[],
        outputs=[input_text, output_label],
    )

main_app = gr.mount_gradio_app(main_app, demo, path="/")
