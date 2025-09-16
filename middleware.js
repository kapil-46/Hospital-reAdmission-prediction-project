module.exports.isauthenticated = (req,res,next)=>{
    //console.log(req.user);
    //console.log(req);
     if(!req.isAuthenticated()){
      req.session.originalurlvar = req.originalUrl;
      req.flash("error","Login required!");
      return res.redirect("/login");
     }
     next();
}

//when we want to add new listing so now we need to login then its redirecting to listing home page but its causing inconvineance to users.
// so to redirect after login to respective page we use this process. where  we are storing the respective url to local variable because when we
// use isAuthenticate function then it will delete all the extra information added to it .because we store that data in a local variable .then we 
//aceess that local var where ever its needed (using path).

module.exports.originalurlfunc = (req,res,next)=>{
    //req.session.originalurlvar upside 
    if(req.session.originalurlvar){
    //local var used in user.js
    res.locals.orurl = req.session.originalurlvar;
    }
    next();
}