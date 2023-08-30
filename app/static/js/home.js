const movingBackground = document.querySelector('.bg-move');
const contentDom = document.querySelector('.content-wrapper');
const titleBtn = document.getElementById('title-button');

window.addEventListener('scroll', () => {
    const dy = -0.5 * window.scrollY;
    movingBackground.style.transform = `translateY(${dy}px)`;
});

titleBtn.addEventListener('click', () => {
    contentDom.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
    });
});