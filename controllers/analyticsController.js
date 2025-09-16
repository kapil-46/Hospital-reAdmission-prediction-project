const Patient = require('../models/Patient');

exports.getKpiData = async (req, res) => {
  try {
    const total = await Patient.countDocuments();

    // High risk defined as readmitted = "yes"
    const highRisk = await Patient.countDocuments({ readmitted: "yes" });

    // Optionally calculate recent month’s new patients
    const now = new Date();
    const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const newThisMonth = await Patient.countDocuments({ createdAt: { $gte: startOfMonth } });

    // For demo, static accuracy; replace with model metric if available
    const accuracy = 94.2;


    const riskLevels = await Patient.aggregate([
      { $group: { _id: "$riskLevel", count: { $sum: 1 } } }
    ]);

    // Last 30 days readmission trends by date
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(now.getDate() - 30);
    const trends = await Patient.aggregate([
      { $match: { createdAt: { $gte: thirtyDaysAgo } } },
      {
        $group: {
          _id: {
            year: { $year: "$createdAt" },
            month: { $month: "$createdAt" },
            day: { $dayOfMonth: "$createdAt" }
          },
          count: { $sum: 1 }
        }
      },
      { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1 } }
    ]);


    // Render and pass only relevant data
    res.render('analytics', {
      total,
      highRisk,
      newThisMonth,
      accuracy,
      riskLevels,
      trends
    });
  } catch (error) {
    console.error('Error fetching KPI data:', error);
    res.status(500).send('Internal Server Error');
  }
};
