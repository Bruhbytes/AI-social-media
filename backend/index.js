const express = require('express');
const bodyParser = require('body-parser');
require('dotenv').config()
const cors = require('cors');


const app = express();
app.use(cors({
    origin: ["https://ai-social-media-client.vercel.app"],
}))
app.use(bodyParser.urlencoded({extended: true}))
app.use(express.json());

app.use((req, res, next) => {
    console.log(req.path, req.method);
    next();
})

app.get('/', (req, res) => {
    res.status(200).json({"msg":"Hello world"});
})

app.listen(4000, (req, res) => {
    console.log("Server started on port 4000");
})