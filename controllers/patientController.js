const axios = require('axios');
const Patient = require('../models/Patient');

const PREDICTION_API_URL = process.env.PREDICTION_API_URL || 'http://localhost:5000';

// Render Input form
exports.getInputForm = (req, res) => {
  if(!req.isAuthenticated()){
    req.flash("error","you must be loggedin to predict");
    return  res.redirect("/login");
  }
  res.render('input');
};

// Handle form submission and create patient entry
exports.submitPatientData = async (req, res) => {
  try {
    const patientData = req.body;

    // Prepare input data for prediction API
    const apiPayload = {
      time_in_hospital: Number(patientData.time_in_hospital),
      n_lab_procedures: Number(patientData.n_lab_procedures),
      n_procedures: Number(patientData.n_procedures),
      n_medications: Number(patientData.n_medications),
      n_outpatient: Number(patientData.n_outpatient),
      n_inpatient: Number(patientData.n_inpatient),
      n_emergency: Number(patientData.n_emergency),
      age: patientData.age,
      medical_specialty: patientData.medical_specialty || 'Missing',
      diag_1: patientData.diag_1 || 'Missing',
      diag_2: patientData.diag_2 || 'Missing',
      diag_3: patientData.diag_3 || 'Missing',
      glucose_test: patientData.glucose_test || 'no',
      A1Ctest: patientData.A1Ctest || 'no',
      change: patientData.change || 'no',
      diabetes_med: patientData.diabetes_med || 'no',
    };

    // Call Flask prediction API
    const response = await axios.post(`${PREDICTION_API_URL}/predict`, apiPayload);
    patientData.readmitted = response.data.readmitted;

    if (response.data.readmitted === "yes") {
    patientData.riskLevel = "High";
} else {
    patientData.riskLevel = "Low";
}

    // Save patient record with real prediction
    const patient = new Patient(patientData);
    await patient.save();

    res.redirect(`/patients/result/${patient._id}`);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).send('Error during prediction');
  }
};

// Show prediction result page for specific patient record
exports.getResultPage = async (req, res) => {
  try {
    const patient = await Patient.findById(req.params.id);
    if (!patient) {
      return res.status(404).send('Patient data not found');
    }
    res.render('result', { patient });
  } catch (error) {
    console.error(error);
    res.status(500).send('Server Error');
  }
};

// List patient history entries
exports.getPatientHistory = async (req, res) => {
   if(!req.isAuthenticated()){
    req.flash("error","you must be loggedin to view histry");
    return  res.redirect("/login");
  }
  try {
    const patients = await Patient.find().sort({ createdAt: -1 }).limit(20);
    res.render('history', { patients });
  } catch (error) {
    console.error(error);
    res.status(500).send('Server Error');
  }
};






