import qai_hub
import torch

import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model
from train import compress_model, GROUP1, GROUP2


class MobCLIP(torch.nn.Module):
    def __init__(self, model_name, weight):
        super(MobCLIP, self).__init__()
        reparameterize = True if weight is None else False
        model, _, _ = mobileclip.create_model_and_transforms(model_name, f'weight/{model_name}.pt',
                                                             reparameterize)

        prun_model = compress_model(model.image_encoder.model, 4, GROUP1, GROUP2)
        if weight is not None:
            prun_model.load_state_dict(torch.load(weight, weights_only=True, map_location='cpu')['state_dict'])
            prun_model = reparameterize_model(prun_model)
        prun_model.eval()

        text_embedding = torch.load(f'weight/text_embedding_mobileclip_s1.pt', weights_only=True)
        self.register_buffer('text_embedding', text_embedding.T)

        self.visual = prun_model

    def forward(self, img):
        with torch.no_grad():
            img = self.visual(img)
        img = torch.nn.functional.normalize(img, dim=-1)
        out = torch.mm(img, self.text_embedding)
        return out


def run_inference(model, device, input_dataset):
    """Submit an inference job for the model."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
    )
    return inference_job.download_output_data()


def run_profile(model, device):
    """Submits a profile job for the model and returns the job ID."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        device=device,
        options="--max_profiler_iterations 100",
        name=''
    )
    return profile_job.download_profile()


def run_compile(model, name, share, device):
    """Submits a compile job for the model."""
    input_shape = (1, 3, 224, 224)
    compile_job = qai_hub.submit_compile_job(
        model,
        name=name,  # Replace with your model name
        device=qai_hub.Device(device),
        input_specs=dict(image=input_shape),
    )
    if share:
        compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])  ## Share your model for submission
    return compile_job.get_target_model()


DEVICE = 'Snapdragon 8 Elite QRD'

if __name__ == '__main__':
    # Create the model
    model = MobCLIP('mobileclip_s1', 'weight/1st_weight.pt')  # 1st

    # Trace model
    example_input = torch.rand(1, 3, 224, 224)
    model.eval()
    pt_model = torch.jit.trace(model, example_input)

    # Compile model on a specific device
    compiled_model = run_compile(pt_model, 'sm_layer_imitation', True, DEVICE)

    # Profile model
    profile_output = run_profile(compiled_model, device=qai_hub.Device(DEVICE))
