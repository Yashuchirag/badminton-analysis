## Backend


python .\data_preparation.py --action annotate --images .\dataset\raw_frames\match1\ --output .\dataset\annotations\match1     

python .\data_preparation.py --action split --images .\dataset\raw_frames\match1 --annotations .\dataset\annotations\match1\ --output .\dataset\processed\match2\ --method rally


python .\train_and_track.py --action train-obb `                                                
>>     --split-dir .\dataset\processed\match2\ `
>>     --output-dir yolo-obb `
>>     --yolo-version 8 `
>>     --epochs 20 `
>>     --batch 8 `
>>     --device 0




python .\train_and_track.py --action track `                                                    
>> --video .\dataset\videos\Sample_3.mp4 `      
>> --output-video Sample_3_tracked.mp4 `
>> --yolo-weights .\runs\detect\yolo-runs\yolo_standard\weights\best.pt --obb-weights .\runs\obb\train-track\yolo_obb\weights\best.pt --tracknet-weights .\runs\train-track\tracknet_best.pth --mode hybrid --device cuda 

uvicorn main:app --host 0.0.0.0 --port 8000 --reload