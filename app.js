require('dotenv').config();
const mongoose = require('mongoose');
const patientRoutes = require('./routes/patientRoutes');
const analyticsRoutes = require('./routes/inde.js');
const userjs = require("./routes/user.js");
const path = require('path');
const { sendMail,userMail } = require("./mailer.js");
const bodyParser = require('body-parser');
const express = require('express');
const app = express();
const passport = require("passport");
const LocalStrategy = require("passport-local").Strategy;
const User = require("./models/users.js");
const MongoStore=require('connect-mongo');


const store=new MongoStore({
mongoUrl:process.env.MONGO_URI,
Crypto:{
secret:"mysupersetcode",
},
touchAfter:24*3600,
  });

   store.on("errror",()=>{
    console.log("error in mongostore",err);
  });

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => console.log('MongoDB Connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// Set view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Middlewares
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));






const session=require("express-session");
const flash=require("connect-flash");
const sessionOptions={
     secret :"musupersecretcode",
     resave : false,
     saveUninitialized:true,
     cookie:{
        expires : Date.now() + 7*24*60*60*1000,
        maxAge : 7*24*60*60*1000,
        httpOnly :true,
     },
};




app.use(session(sessionOptions));
app.use(flash());

app.use(passport.initialize());
app.use(passport.session());

// use static authenticate method of model in LocalStrategy
passport.use(new LocalStrategy(User.authenticate()));

// use static serialize and deserialize of model for passport session support
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());







app.use((req,res,next)=>{
     res.locals.successcreating= req.flash("success");
     res.locals.error= req.flash("error");
     res.locals.curruser= req.user;
     next();
});
// Routes
app.get('/', (req, res) => {
  res.render('home');
});



app.use('/patients', patientRoutes);
app.use('/', analyticsRoutes);
app.use("/",userjs);


app.get('/about', (req, res) => {
  res.render('about');
});
app.get('/faq', (req, res) => {
  res.render('faq');
});
app.get('/contact', (req, res) => {
  res.render('contact');
});

app.post('/contact', async(req, res) => {
  
  const {name,email,message}=req.body;
  console.log(name,email,message);
  await sendMail({
  name:name,
  email:email,
 message:message
});

await userMail({
  email:email
});
req.flash("success","Message Sent Successfully");
  // You can later save this info to DB or send email
  // res.send('Thank you for contacting us!');
});


app.get('/analytics', (req, res) => {
  res.render('analytics');
});

// 404 Handler
app.use((req, res) => {
  res.status(404).send('Page Not Found');
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0',() => {
  console.log(`Server running on port ${PORT}`);
});
