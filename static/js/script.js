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

// Fake analysis and populate results
const form = document.getElementById('analysisForm');
const submitBtn = document.getElementById('submitBtn');
if (form && submitBtn){
  form.addEventListener('submit', (e)=>{
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing…';

    setTimeout(()=>{
      const confidence = (85 + Math.floor(Math.random()*10)) + '%';
      document.getElementById('confidence').textContent = confidence;
      document.getElementById('riskPill').textContent = 'Low Risk';
      // mark checklist visually (already checked via CSS)
      // scroll to results
      document.getElementById('results').scrollIntoView({behavior:'smooth'});
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Analysis';
    }, 1200);
  });
}

const reanalyzeBtn = document.getElementById('reanalyzeBtn');
if (reanalyzeBtn){
  reanalyzeBtn.addEventListener('click', ()=>{
    document.getElementById('analyze').scrollIntoView({behavior:'smooth'});
  });
}
