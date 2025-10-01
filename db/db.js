const mongoose = require('mongoose');
const dotenv = require('dotenv');
dotenv.config();

async function dbConnect() {
    let dbUrl = process.env.db_url;

    await mongoose.connect(dbUrl).then(()=>{
        console.log("Database connected successfully");
    }).catch((err)=>{
        console.log("Database connection failed");
        console.log(err);
    });

}

module.exports = dbConnect;