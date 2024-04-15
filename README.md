# CS5242
## Front end

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

## Back end

To run back end application to listen the http request from fronted, use python and uvicorn.

```shell
cd backend
uvicorn backend.asgi:application
```


