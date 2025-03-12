import base64
import gradio as gr
import json
from typing import List, Dict, Any, Tuple

from annotator.openpose import decode_json_as_poses, draw_poses
from annotator.openpose.animalpose import draw_animalposes
from lib_controlnet.controlnet_ui.modal import ModalInterface
from modules import shared
from lib_controlnet.logging import logger


def parse_data_url(data_url: str):
    # Split the URL at the comma
    media_type, data = data_url.split(",", 1)

    # Check if the data is base64-encoded
    assert ";base64" in media_type

    # Decode the base64 data
    return base64.b64decode(data)


def encode_data_url(json_string: str) -> str:
    base64_encoded_json = base64.b64encode(json_string.encode("utf-8")).decode("utf-8")
    return f"data:application/json;base64,{base64_encoded_json}"


class OpenposeEditor(object):
    # Filename used when user click the download link.
    download_file = "pose.json"
    # URL the openpose editor is mounted on.
    editor_url = "/openpose_editor_index"

    def __init__(self) -> None:
        self.render_button = None
        self.pose_input = None
        self.download_link = None
        self.upload_link = None
        self.modal = None

    def render_edit(self):
        """Renders the buttons in preview image control button group."""
        # The hidden button to trigger a re-render of generated image.
        self.render_button = gr.Button(visible=False, elem_classes=["cnet-render-pose"])
        # The hidden element that stores the pose json for backend retrieval.
        # The front-end javascript will write the edited JSON data to the element.
        self.pose_input = gr.Textbox(visible=False, elem_classes=["cnet-pose-json"])

        self.modal = ModalInterface(
            # Use about:blank here as placeholder so that the iframe does not
            # immediately navigate. Most of controlnet units do not need
            # openpose editor active. Only navigate when the user first click
            # 'Edit'. The navigation logic is in `openpose_editor.js`.
            f'<iframe src="about:blank"></iframe>',
            open_button_text="Edit",
            open_button_classes=["cnet-edit-pose"],
            open_button_extra_attrs=f'title="Send pose to {OpenposeEditor.editor_url} for edit."',
        ).create_modal(visible=False)
        self.download_link = gr.HTML(
            value=f"""<a href='' download='{OpenposeEditor.download_file}'>JSON</a>""",
            visible=False,
            elem_classes=["cnet-download-pose"],
        )

    def render_upload(self):
        """Renders the button in input image control button group."""
        self.upload_link = gr.HTML(
            value="""
            <label>Upload JSON</label>
            <input type="file" accept=".json"/>
            """,
            visible=False,
            elem_classes=["cnet-upload-pose"],
        )

    def register_callbacks(
        self,
        generated_image: gr.Image,
        use_preview_as_input: gr.Checkbox,
        model: gr.Dropdown,
    ):
        def render_pose(pose_url: str) -> Tuple[Dict, Dict]:
            json_string = parse_data_url(pose_url).decode("utf-8")
            poses, animals, height, width = decode_json_as_poses(
                json.loads(json_string)
            )
            logger.info("Preview as input is enabled.")
            return (
                # Generated image.
                gr.update(
                    value=(
                        draw_poses(
                            poses,
                            height,
                            width,
                            draw_body=True,
                            draw_hand=True,
                            draw_face=True,
                        )
                        if poses
                        else draw_animalposes(animals, height, width)
                    ),
                    visible=True,
                ),
                # Use preview as input.
                gr.update(value=True),
                # Self content.
                *self.update(json_string),
            )

        self.render_button.click(
            fn=render_pose,
            inputs=[self.pose_input],
            outputs=[generated_image.background, use_preview_as_input, *self.outputs()],
        )

        def update_upload_link(model: str) -> Dict:
            return gr.update(visible="openpose" in model.lower())

        model.change(fn=update_upload_link, inputs=[model], outputs=[self.upload_link])

    def outputs(self) -> List[Any]:
        return [
            self.download_link,
            self.modal,
        ]

    def update(self, json_string: str) -> List[Dict]:
        """
        Called when there is a new JSON pose value generated by running
        preprocessor.

        Args:
            json_string: The new JSON string generated by preprocessor.

        Returns:
            An gr.update event.
        """
        hint = "Download the pose as .json file"
        html = f"""<a href='{encode_data_url(json_string)}' 
                      download='{OpenposeEditor.download_file}' title="{hint}">
                    JSON</a>"""

        visible = json_string != ""
        return [
            # Download link update.
            gr.update(value=html, visible=visible),
            # Modal update.
            gr.update(
                visible=visible
                and not shared.opts.data.get("controlnet_disable_openpose_edit", False)
            ),
        ]
