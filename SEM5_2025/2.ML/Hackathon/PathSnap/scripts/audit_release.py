from pathlib import Path
import sys,json,hashlib,zipfile,importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT,save_json
checks={}
for name in ("Student folder.zip","PathSnap_student_materials.zip"):
    archive=ROOT.parent/name
    with zipfile.ZipFile(archive) as z:
        count=0
        for entry in z.infolist():
            if entry.is_dir(): continue
            original=ROOT.parent/entry.filename if name=="Student folder.zip" else ROOT.parent/"PathSnap_student_materials"/entry.filename
            assert original.read_bytes()==z.read(entry),f"Original changed: {original}"
            count+=1
    checks[name]={"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"verified_original_files":count}
required=["src/build_features.py","src/train_models.py","src/tune_model.py","src/hmm_viterbi.py","app/api.py","ui/app.py","README.md","requirements.txt",
    "artifacts/best_model.joblib","configs/deployment.json","results/models/comparison.csv","results/tuning/trials.csv","results/hmm/summary.json","results/logs/http_smoke.json","results/logs/ui_smoke.json"]
for name in required: assert (ROOT/name).stat().st_size>0,name
assert len(list((ROOT/"data/train").iterdir()))==80
assert len(list((ROOT/"data/student_test").iterdir()))==10
assert len(list((ROOT/"data/evaluator_test").iterdir()))==0
save_json(ROOT/"results/verification.json",dict(required_files_verified=required,original_inputs=checks,evaluator_data_used=False,cross_laptop_verified=False,github_operations_performed=False,technical_report_created=False))
packages=["numpy","pandas","scipy","scikit-learn","joblib","threadpoolctl","matplotlib","optuna","fastapi","uvicorn","streamlit","folium","requests","httpx","pytest"]
save_json(ROOT/"configs/runtime_versions.json",{p:importlib.metadata.version(p) for p in packages})
manifest=[]
for folder in ["src","app","ui","scripts","tests","configs"]:
    for p in (ROOT/folder).rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
save_json(ROOT/"results/source_manifest.json",manifest)
print("Original ZIP contents verified; all required deliverable files present; evaluator data absent.")

