const mongoose = require('mongoose');

const patientSchema = new mongoose.Schema({
  age: String,
  time_in_hospital: Number,
  n_lab_procedures: Number,
  n_procedures: Number,
  n_medications: Number,
  n_outpatient: Number,
  n_inpatient: Number,
  n_emergency: Number,
  medical_specialty: String,
  diag_1: String,
  diag_2: String,
  diag_3: String,
  glucose_test: String,
  A1Ctest: String,
  change: String,
  diabetes_med: String,
  readmitted: String, 
  riskLevel: String, // Store prediction result ("yes" or "no")
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('Patient', patientSchema);
