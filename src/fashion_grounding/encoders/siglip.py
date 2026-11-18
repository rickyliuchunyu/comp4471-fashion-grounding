"""Pinned SigLIP 2 FixRes using official preprocessing and frozen standard outputs."""
import importlib.metadata as metadata
from pathlib import Path
import re
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from fashion_grounding.data.schema import require, read_json, sha256
from fashion_grounding.grounding.cache import check_features

class SiglipEncoder:
    def __init__(self, snapshot, revision, device="cpu", batch_size=8):
        require(bool(re.fullmatch(r"[0-9a-f]{40}",revision)), "immutable model revision required")
        require(batch_size > 0, "positive batch size required")
        self.snapshot=Path(snapshot)
        manifest=read_json(self.snapshot/"download_manifest.json")
        require(manifest["revision"]==revision, "snapshot revision mismatch")
        require(manifest["model_id"]=="google/siglip2-base-patch16-224", "unexpected encoder")
        for name,item in manifest["files"].items():
            require(sha256(self.snapshot/name)==item["sha256"], "snapshot bytes changed: "+name)
        self.device=torch.device(device)
        self.batch_size=batch_size
        self.processor=AutoProcessor.from_pretrained(snapshot,local_files_only=True,use_fast=False)
        self.model=AutoModel.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,
                                            torch_dtype=torch.float32,attn_implementation="eager").to(self.device)
        self.model.eval().requires_grad_(False)
        self.dimension=int(self.model.config.text_config.hidden_size)
        self.max_length=int(self.model.config.text_config.max_position_embeddings)
        self.logit_scale=float(self.model.logit_scale.detach().cpu().item())
        self.logit_bias=float(self.model.logit_bias.detach().cpu().item())
        self.signature={"model_id":manifest["model_id"],"revision":revision,
                        "files":{k:v["sha256"] for k,v in manifest["files"].items()},
                        "processor_class":type(self.processor).__name__,
                        "versions":{p:metadata.version(p) for p in ("transformers","tokenizers","torch","torchvision","Pillow","sentencepiece","numpy")},
                        "dtype":"float32","attention":"eager","device":str(self.device),
                        "max_length":self.max_length,"dimension":self.dimension,
                        "logit_scale":self.logit_scale,"logit_bias":self.logit_bias,
                        "padding":"max_length","truncation":False}

    def _text_inputs(self,texts):
        require(all(isinstance(t,str) and t.strip() for t in texts), "empty text")
        # Same tokenizer, including its official normalization and special tokens.
        tokens=self.processor.tokenizer(texts,padding=False,truncation=False)
        require(all(len(ids)<=self.max_length for ids in tokens["input_ids"]), "query exceeds model context; review it instead of truncating")
        return self.processor(text=texts,padding="max_length",max_length=self.max_length,
                              truncation=False,return_tensors="pt")

    def _normalized(self,tensor):
        require(bool(torch.isfinite(tensor).all()), "nonfinite encoder output")
        norms=tensor.norm(dim=-1,keepdim=True)
        require(bool((norms>0).all()), "zero encoder output")
        values=(tensor/norms).detach().cpu().numpy().astype(np.float32)
        return check_features(values,len(values),self.dimension)

    def encode_images(self,images):
        result=[]
        with torch.inference_mode():
            for start in range(0,len(images),self.batch_size):
                inputs=self.processor(images=images[start:start+self.batch_size],return_tensors="pt").to(self.device)
                result.append(self._normalized(self.model.get_image_features(**inputs)))
        return np.concatenate(result) if result else np.empty((0,self.dimension),np.float32)

    def encode_texts(self,texts):
        result=[]
        with torch.inference_mode():
            for start in range(0,len(texts),self.batch_size):
                inputs=self._text_inputs(texts[start:start+self.batch_size]).to(self.device)
                result.append(self._normalized(self.model.get_text_features(**inputs)))
        return np.concatenate(result) if result else np.empty((0,self.dimension),np.float32)

    def verify_logits(self,images,texts):
        visual=self.encode_images(images); textual=self.encode_texts(texts)
        inputs=dict(self.processor(images=images,return_tensors="pt"))
        inputs.update(self._text_inputs(texts))
        with torch.inference_mode():
            standard=self.model(**{k:v.to(self.device) for k,v in inputs.items()}).logits_per_text.cpu().numpy()
        rebuilt=np.exp(self.logit_scale)*(textual@visual.T)+self.logit_bias
        require(np.allclose(standard,rebuilt,rtol=2e-4,atol=2e-4), "reconstructed logits differ")
        require(not self.model.training and not any(p.requires_grad for p in self.model.parameters()), "encoder not frozen")
        return {"dimension":self.dimension,"max_abs_logit_error":float(np.max(np.abs(standard-rebuilt))),
                "frozen":True,"images":len(images),"texts":len(texts)}
