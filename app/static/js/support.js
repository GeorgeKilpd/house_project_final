document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------
    // 1) 카드 필터링
    // -----------------------------
    const checkboxes = document.querySelectorAll(".btn-check");
    const resetBtn = document.getElementById("reset-filters");

    function filterCards() {
        const activeFilters = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        const allCards = document.querySelectorAll(".sp-card");

        allCards.forEach(card => {
            const cardType = card.dataset.biz;
            // 필터 없으면 전체, 있으면 포함되는 카드만 보여주기
            if (activeFilters.length === 0 || activeFilters.includes(cardType)) {
                card.closest(".swiper-slide").style.display = "block";
            } else {
                card.closest(".swiper-slide").style.display = "none";
            }
        });

        // Swiper 업데이트 필요
        document.querySelectorAll(".swiper").forEach(swiperEl => {
            if (swiperEl.swiper) swiperEl.swiper.update();
        });
    }

    checkboxes.forEach(cb => {
        cb.addEventListener("change", filterCards);
    });

    // -----------------------------
    // 2) 초기화 버튼
    // -----------------------------
    resetBtn.addEventListener("click", () => {
        checkboxes.forEach(cb => cb.checked = false);
        filterCards();
    });

    // -----------------------------
    // 3) Swiper 초기화
    // -----------------------------
    const sections = ["청년", "신혼부부"];
    sections.forEach(section => {
        const swiperEl = document.querySelector(`.sp-swiper-${section.toLowerCase()}`);
        new Swiper(swiperEl, {
            slidesPerView: 'auto',
            spaceBetween: 12,
            navigation: {
                nextEl: `.sp-next-${section.toLowerCase()}`,
                prevEl: `.sp-prev-${section.toLowerCase()}`,
            },
            breakpoints: {
                480: { slidesPerView: 1.2, spaceBetween: 10 },
                576: { slidesPerView: 2, spaceBetween: 10 },
                768: { slidesPerView: 2.5, spaceBetween: 12 },
                992: { slidesPerView: 3, spaceBetween: 12 }
            }
        });
    });

    // -----------------------------
    // 4) 초기 필터 적용
    // -----------------------------
    filterCards();
});
