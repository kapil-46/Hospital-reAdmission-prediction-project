const User=require("../models/users.js");
const { sendVerificationMail } = require("../mailer.js");



module.exports.rendersignup=async(req,res)=>{
        res.render("signup");
}


module.exports.postsignup =async(req,res)=>{

        try{
        const {username,email,password} = req.body;
        const otp = Math.floor(100000 + Math.random() * 900000).toString(); // 6-digit
        const otpExpiry = Date.now() + 3* 60 * 1000;      




        const newuser = new User({username,email,otp,otpExpiry,isVerified: false});
        const k=await User.register(newuser,password);
        



        await sendVerificationMail({email, otp});

        //  req.flash("success", "OTP sent to your email. Please verify.");
         res.redirect(`/verify/${k._id}`);
          
         //shifted to 
//         req.login(k,(err)=>{
//         if(err){
//         return next(err);
//         }
//        req.flash("success","welcome to wanderlust");
//         res.redirect("/");

       // }
//);
        }catch(e){
         if (e.code === 11000) {
  req.flash("error", "Email already exists.");
 
} else {
  req.flash("error", e.message);
 
}
     res.redirect("signup");           
        }

        
}

module.exports.renderlogin =async(req,res)=>{
        res.render("login");
}


module.exports.postlogin =async(req,res)=>{
           req.flash("success","welcome back to Hospital Readmission Predictor");
         let m=res.locals.orurl || "/";
        res.redirect(m);
}

module.exports.logout=(req,res,next)=>{
        req.logout((err)=>{
        if(err){
        return next(err);
        }
        req.flash("success","you are logged out!");
        res.redirect("/");

        });
}
