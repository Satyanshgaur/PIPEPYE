/**
 * PipePye Research Report — Client Script
 * Zero-dependency UI interactions for GitHub Pages
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Scroll-spy for Top Nav and TOC
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a, .toc-nav a');

  const onScroll = () => {
    const scrollPosition = window.scrollY + 140;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      const id = section.getAttribute('id');

      if (scrollPosition >= top && scrollPosition < top + height) {
        navLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // 2. Interactive Filter for Industrial Benchmark Ladder
  const filterButtons = document.querySelectorAll('[data-bench-filter]');
  const benchRows = document.querySelectorAll('[data-bench-class]');

  if (filterButtons.length > 0) {
    filterButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const filter = btn.getAttribute('data-bench-filter');

        filterButtons.forEach(b => {
          b.classList.remove('active');
          b.style.backgroundColor = '#fafafa';
          b.style.color = 'var(--color-carbon)';
        });
        btn.classList.add('active');
        btn.style.backgroundColor = 'var(--color-carbon)';
        btn.style.color = '#ffffff';

        benchRows.forEach(row => {
          const rowClass = row.getAttribute('data-bench-class');
          if (filter === 'all' || rowClass === filter) {
            row.style.display = '';
          } else {
            row.style.display = 'none';
          }
        });
      });
    });
  }

  // 3. Smooth Anchor Scrolling Offset
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);
      if (targetElement) {
        e.preventDefault();
        const headerOffset = 64;
        const elementPosition = targetElement.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    });
  });
});
