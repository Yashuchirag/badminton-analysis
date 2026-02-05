## Backend


python .\data_preparation.py --action annotate --images .\dataset\raw_frames\match1\ --output .\dataset\annotations\match1     

python .\data_preparation.py --action split --images .\dataset\raw_frames\match1 --annotations .\dataset\annotations\match1\ --output .\dataset\processed\match2\ --method rally



uvicorn main:app --host 0.0.0.0 --port 8000 --reload