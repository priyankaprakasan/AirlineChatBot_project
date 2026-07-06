from huggingface_hub import HfApi
import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="deployment",     # the local folder containing your files
    repo_id="priyankaprakasan/HRchatBot",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
