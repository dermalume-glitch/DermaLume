// Mobile nav toggle
const toggle = document.querySelector('.nav-toggle');
const menu = document.getElementById('nav-menu');
if (toggle && menu) {
  toggle.addEventListener('click', () => {
    const open = menu.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
}

// Footer year
const year = document.getElementById('year');
if (year) year.textContent = new Date().getFullYear();

// Simple CTA form demo
const form = document.getElementById('cta-form');
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = form.querySelector('#email').value.trim();
    const msg = form.querySelector('.form__msg');
    if (!email) {
      msg.textContent = 'Please enter a valid email.';
      msg.style.color = 'salmon';
      return;
    }
    msg.textContent = 'Thanks! We will be in touch.';
    msg.style.color = '#59e1a5';
    form.reset();
  });
}
