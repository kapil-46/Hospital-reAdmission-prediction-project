const express = require('express');
const router = express.Router();
const patientController = require('../controllers/patientController');

router.get('/input', patientController.getInputForm);
router.post('/input', patientController.submitPatientData);
router.get('/result/:id', patientController.getResultPage);
router.get('/history', patientController.getPatientHistory);






module.exports = router;
