const nodemailer = require("nodemailer");

const transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS
  }
});

module.exports.sendMail = async ({ name, email, message }) => {
    console.log("hii");
    await transporter.sendMail({
    from: email,
    to:process.env.SMTP_USER,
    subject: `message from ${email}`,
    html: `
    <p>${name}</p>
      <p>${message}</p>
    `,
  });
};


module.exports.userMail = async ({email}) => {
    await transporter.sendMail({
    from:process.env.SMTP_USER,
    to:email,
    subject: `message sent`,
    html: `
    <p>We received your message and proceesing your feedback,your issue will be resolved soon.Thank you for contacting us.</p>
    `,
  });
};



module.exports.sendVerificationMail=async({email,otp})=>{await transporter.sendMail({
    from: process.env.SMTP_USER,
    to:   email,
    subject: "Verify your account",
    html: `<h2>${otp}</h2><p>This code expires in 2 minutes.</p>`
  });
};