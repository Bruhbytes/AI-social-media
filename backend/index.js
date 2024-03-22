const express = require('express');
const bodyParser = require('body-parser');
require('dotenv').config()
const passport = require("passport");
const cookieSession = require("cookie-session");
const passportStrategy = require("./passport");
const authRoute = require("./routes/auth");
const cors = require('cors');

const frontendUrl = "https://ai-social-media-client.vercel.app";


const app = express();
app.use(cors({
    origin: ["http://localhost:3000"],    
    // origin: [frontendUrl],
    methods: "GET,POST,PUT,DELETE",
    credentials: true,
}))
app.use(bodyParser.urlencoded({ extended: true }))
app.use(express.json());
app.use(
    cookieSession({
        name: "session",
        keys: ["cyberwolve"],
        maxAge: 24 * 60 * 60 * 100,
    })
);
app.use(passport.initialize());
app.use(passport.session());



app.use((req, res, next) => {
    console.log(req.path, req.method);
    next();
})

app.use("/auth", authRoute);

app.get('/', (req, res) => {
    res.status(200).json({ "msg": "Hello world" });
})

app.listen(4000, (req, res) => {
    console.log("Server started on port 4000");
})