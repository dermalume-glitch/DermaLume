// Basic interactivity for DermaLume demo
const navLinks = document.querySelectorAll('.nav-link');
navLinks.forEach(a => {
  a.addEventListener('click', (e) => {
    // smooth scroll
    const href = a.getAttribute('href');
    if (href?.startsWith('#')){
      e.preventDefault();
      document.querySelector(href)?.scrollIntoView({behavior:'smooth'});
    }
  });
});

// Upload preview
const dropzone = document.getElementById('dropzone');
const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
if (dropzone && imageInput && preview){
  const openFile = () => imageInput.click();
  dropzone.addEventListener('click', openFile);
  dropzone.addEventListener('keydown', (e)=>{ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); openFile(); }});
  imageInput.addEventListener('change', () => {
    const file = imageInput.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      preview.innerHTML = `<img src="${reader.result}" alt="Uploaded spot" style="max-width:100%; border-radius:12px;">`;
      // also mirror into results
      const resultPreview = document.getElementById('resultPreview');
      if(resultPreview){
        resultPreview.style.background = `center/cover no-repeat url('${reader.result}')`;
        resultPreview.style.border = '1px solid #eadfdb';
      }
    };
    reader.readAsDataURL(file);
  });
}

// Real API analysis
const form = document.getElementById('analysisForm');
const submitBtn = document.getElementById('submitBtn');
if (form && submitBtn){
  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    
    // Get form data
    const imageFile = document.getElementById('imageInput').files?.[0];
    const age = document.getElementById('age').value;
    const gender = document.querySelector('input[name="gender"]:checked')?.value;
    const location = document.getElementById('location').value;
    
    if (!imageFile || !age || !gender || !location) {
      alert('Please fill all fields and upload an image');
      return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('age', age);
    formData.append('gender', gender);
    formData.append('location', location);
    
    console.log('Sending data:', {
      image: imageFile.name,
      age: age,
      gender: gender,
      location: location
    });
    
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing…';

    try {
      // Call the API
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
      });
      
      // Check if response is ok before parsing JSON
      const responseText = await response.text();
      console.log('Response status:', response.status);
      console.log('Response text:', responseText);
      
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        console.error('Failed to parse response:', e);
        throw new Error('Server returned invalid response');
      }
      
      if (response.ok) {
        // Update UI with results
        const confidence = Math.round(result.confidence * 100) + '%';
        document.getElementById('confidence').textContent = confidence;
        
        // Map diagnosis to risk level
        const riskMapping = {
          'mel': 'High Risk - Cancerous',      // Melanoma
          'bcc': 'High Risk - Cancerous',      // Basal cell carcinoma
          'akiec': 'Medium Risk - Pre-cancerous', // Actinic keratoses
          'bkl': 'Low Risk - Benign',          // Benign keratosis
          'nv': 'Low Risk - Benign',           // Common moles
          'df': 'Low Risk - Benign',           // Dermatofibroma
          'vasc': 'Low Risk - Benign'          // Vascular lesions
        };
        
        const riskLevel = riskMapping[result.prediction] || 'Unknown Risk';
        const riskPill = document.getElementById('riskPill');
        riskPill.textContent = riskLevel;
        
        // Update risk pill color
        riskPill.className = 'pill ' + (riskLevel.includes('High') ? 'high' : riskLevel.includes('Medium') ? 'medium' : '');
        
        // Update diagnosis text
        document.querySelector('.result-heading').textContent = `${riskLevel}: ${result.dx_full || result.prediction}`;
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({behavior:'smooth'});
      } else {
        alert('Error: ' + (result.error || 'Unknown error occurred'));
      }
    } catch (error) {
      alert('Error connecting to server: ' + error.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Analysis';
    }
  });
}

const reanalyzeBtn = document.getElementById('reanalyzeBtn');
if (reanalyzeBtn){
  reanalyzeBtn.addEventListener('click', ()=>{
    document.getElementById('analyze').scrollIntoView({behavior:'smooth'});
  });
}
