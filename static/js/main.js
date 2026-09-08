const PRESETS = {
  scholar: {
    gender: "Male",
    study_time_hours: 5.9,
    attendance_percent: 100.0,
    sleep_hours: 7.3,
    parental_education: "Bachelors",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "No",
    previous_grade: 89.8,
  },
  turnaround: {
    gender: "Male",
    study_time_hours: 4.6,
    attendance_percent: 100.0,
    sleep_hours: 5.8,
    parental_education: "Masters",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "No",
    previous_grade: 37.5,
  },
  at_risk: {
    gender: "Female",
    study_time_hours: 1.9,
    attendance_percent: 69.0,
    sleep_hours: 6.0,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "No",
    part_time_job: "Yes",
    previous_grade: 34.5,
  },
  multitasker: {
    gender: "Female",
    study_time_hours: 6.3,
    attendance_percent: 100.0,
    sleep_hours: 5.7,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "Yes",
    previous_grade: 75.5,
  },
  passive: {
    gender: "Male",
    study_time_hours: 1.5,
    attendance_percent: 100.0,
    sleep_hours: 5.0,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "Yes",
    part_time_job: "Yes",
    previous_grade: 58.6,
  },
  baseline: {
    gender: "Male",
    study_time_hours: 3.6,
    attendance_percent: 85.0,
    sleep_hours: 6.8,
    parental_education: "High School",
    internet_access: "Yes",
    extracurricular_activities: "No",
    part_time_job: "No",
    previous_grade: 69.7,
  },
};

function applyPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  // 1. Synchronize HTML form elements with preset values
  Object.keys(preset).forEach((key) => {
    const field = document.getElementById(key);
    if (field) {
      field.value = preset[key];

      // Dispatch change event to sync linked slider/number labels if present
      field.dispatchEvent(new Event("input", { bubbles: true }));
      field.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });

  // 2. Automatically invoke the prediction request pipeline
  if (typeof runPrediction === "function") {
    runPrediction();
  } else {
    // Fallback: Submit form directly if runPrediction isn't defined globally
    const predictionForm = document.getElementById("predictionForm");
    if (predictionForm) {
      predictionForm.dispatchEvent(
        new Event("submit", { cancelable: true, bubbles: true }),
      );
    }
  }
}
