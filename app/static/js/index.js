// index.js
document.addEventListener('DOMContentLoaded', function () {
    // Hero Carousel 초기화
    const heroCarousel = document.querySelector('#heroCarousel');
    if (heroCarousel) {
        const carousel = new bootstrap.Carousel(heroCarousel, {
            interval: 3500,   // 3.5초마다 자동 슬라이드
            ride: 'carousel',  // 페이지 로드 시 자동 시작
            pause: false,      // 마우스 오버 시 멈추지 않음
            wrap: true         // 마지막 슬라이드 후 다시 처음으로
        });
    }

    // Swiper 초기화
    const policySwiper = new Swiper(".myPolicySwiper", {
        slidesPerView: 3,
        slidesPerGroup: 3,
        spaceBetween: 32,
        navigation: {
            nextEl: ".policy-next",
            prevEl: ".policy-prev",
        },
        grabCursor: true,
        speed: 600,
    });

    // 추후 다른 JS 기능 추가 가능
});
