const express = require("express");
const app = express();

const fs = require("fs");

app.use(express.static("public"));

app.get("/colors", (req, res) => {
  fs.readFile("./everycolorbot.json", (err, json) => {
    let obj = JSON.parse(json);
    res.json(obj);
  });
});

const port = 3333;

app.listen(port, () => console.log(`Example app listening on port ${port}!`));
