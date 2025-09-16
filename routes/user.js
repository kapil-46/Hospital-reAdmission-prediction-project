const express = require('express');
const router = express.Router({mergeParams:true});
const passport=require("passport");
const usercontroller =require("../controllers/userss.js");
const {originalurlfunc}=require("../middleware.js");
const User = require("../models/users.js");

router.get("/signup",usercontroller.rendersignup);
router.post("/signup",usercontroller.postsignup);

router.get("/login",usercontroller.renderlogin);


//4
router.post("/login",originalurlfunc,passport.authenticate('local', { 
        failureRedirect: '/login',
        failureFlash :true

}),usercontroller.postlogin);


router.get("/verify/:id", (req, res) => {
  res.render("verify", { userId: req.params.id });
});




router.post("/verify/:id", async (req, res, next) => {
  const { id } = req.params;
  const { otp} = req.body;
  

  try {
    const user = await User.findById(id);
    
    
    if (!user) {
      req.flash("error", "User not found.");
      return res.redirect("signup");
    }

    if (user.otp !== otp || Date.now() > user.otpExpiry) {
      req.flash("error", "Invalid or expired OTP.");
      return res.redirect(`/verify/${id}`);
    }

    // OTP 
  
    user.isVerified = true;
    user.otp = undefined;
    user.otpExpiry = undefined;
    await user.save();

    // Log the user in after verification
    req.login(user, (err) => {
      if (err) return next(err);
      req.flash("success", "Welcome to Hospital ReAdmission Predictor");
      res.redirect("/");
      
    });

  } catch (e) {
    console.error("OTP Verification Error:", e);
    req.flash("error", "Something went wrong during verification.");
    res.redirect("signup");
  }
});


router.get("/logout",usercontroller.logout);


module.exports = router;