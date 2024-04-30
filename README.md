# CS5242

## Installation

```bash
conda env create -f environment.yml
```

Download the models to the root of project directory like this:
```
models/
┣ all/
┃ ┗ model_e100_lr0.0001.pt.100
┣ math/
┃ ┗ model_e100_lr0.0001.pt.100
┣ molecules/
┃ ┗ model_e100_lr0.0001.pt.100
┣ music/
┃ ┗ model_e100_lr0.0001.pt.100
┗ tables/
  ┗ model_e100_lr0.0001.pt.100
```

* Math: [data](https://huggingface.co/datasets/yuntian-deng/im2latex-100k) [model](models/math/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Simple Tables: [data](https://huggingface.co/datasets/yuntian-deng/im2html-100k) [model](models/tables/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Sheet Music: [data](https://huggingface.co/datasets/yuntian-deng/im2ly-35k-syn) [model](music/math/scheduled_sampling/model_e100_lr0.0001.pt.100)
* Molecules: [data](https://huggingface.co/datasets/yuntian-deng/im2smiles-20k) [model](models/molecules/scheduled_sampling/model_e100_lr0.0001.pt.100)
* End-to-End: [model](https://drive.google.com/drive/folders/1v1d-jrByI84gSve-pfITU9Ij75PGtcOz?usp=drive_link)

## Launch

### Front end

To run front end application, use node.js and npm:

```shell
cd frontend/mark2image-fronted
npm run install
npm run dev
```

After run, you can input the address:

```shell
localhost:8080
```

### Back end

To run back end application to listen the http request from fronted, use python and uvicorn.

```shell
cd backend
uvicorn backend.asgi:application
```

## Train

To run on cluster, please check `sh_scripts` folder.

To run End-to-End model on terminal,
```bash
python src/train.py --dataset_name all --save_dir models/all
```
