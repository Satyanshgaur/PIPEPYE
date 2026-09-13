/**
 * PipePye Research Report — Client Script
 * Zero-dependency UI interactions for GitHub Pages
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Scroll-spy for Top Nav and TOC
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a, .toc-nav a');

  const onScroll = () => {
    const scrollPosition = window.scrollY + 120;

    sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      const id = section.getAttribute('id');

      if (scrollPosition >= top && scrollPosition < top + height) {
        navLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.style.opacity = '1';
            link.style.fontWeight = '700';
          } else {
            link.style.opacity = '0.7';
            link.style.fontWeight = '400';
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
          b.style.backgroundColor = 'transparent';
          b.style.color = 'inherit';
        });
        btn.style.backgroundColor = 'currentColor';
        btn.style.color = btn.closest('.canvas-carbon') ? '#111' : '#fff';

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
