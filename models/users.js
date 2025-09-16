const mongoose=require("mongoose");
const Schema= mongoose.Schema;
const passportLocalMongoose=require("passport-local-mongoose");

const userSchema=new Schema({
         email :{
            type:String,
            required:true,
             unique: true
         },
        //  number :{
        //   type:Number,
        //   requried:true,
        //   unique:true,
        //  },
         isVerified: {
    type: Boolean,
    default: false
  },
  otp: String,           // stores the 6-digit code
  otpExpiry: Date        // when the code expires
 });

 userSchema.plugin(passportLocalMongoose);

 module.exports=mongoose.model("User",userSchema);