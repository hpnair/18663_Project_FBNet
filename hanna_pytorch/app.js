var express = require('express');
var bodyParser = require('body-parser');
const { spawn } = require('child_process');
var path = require('path');


const port = 8080;

var app = express();
var update = "";


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: false}));

app.use(express.static(path.join(__dirname, '/')));


app.get('/', (req, res)=>{
    res.render('index.html');
});

app.post('/', (req, res)=>{

  var submit = req.body.submit_string;
  var splitString = submit.split(" ");

//spawn(splitString[0],splitString.slice(1,22), { stdio: 'inherit' });
const run = spawn(splitString[0],splitString.slice(1,22));

run.stdout.on('data',(data) => {

console.log(`${data}`);
update = data;

});

});

app.put('/',(req,res)=>{

//console.log(update);
res.send(update);

});

app.listen(port , ()=> {
    console.log(`Backend server is listening at port ${port}`);
});























