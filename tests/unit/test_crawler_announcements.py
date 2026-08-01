from bs4 import BeautifulSoup

from app.services.crawler_service import CrawlerService


def test_announcement_listing_selects_first_10_valid_links():
    service = CrawlerService.__new__(CrawlerService)
    html = (
        '<nav><a href="/site/background">Background</a></nav>'
        + "".join(
            f'<a href="/item/{index}">Announcement {index}</a>'
            for index in range(1, 13)
        )
    )
    soup = BeautifulSoup(html, "html.parser")

    links = service._extract_recent_listing_links(
        soup,
        "https://www.udom.ac.tz/announcements",
        10,
        ["www.udom.ac.tz"],
        [],
    )

    assert len(links) == 10
    assert links[0] == ("https://www.udom.ac.tz/item/1", "Announcement 1")
    assert links[-1] == ("https://www.udom.ac.tz/item/10", "Announcement 10")


def test_blog_listing_limit_is_5_and_does_not_require_blog_keyword():
    service = CrawlerService.__new__(CrawlerService)
    html = "".join(
        f'<a href="/posts/{index}">Blog {index}</a>'
        for index in range(1, 8)
    )
    soup = BeautifulSoup(html, "html.parser")

    limit = service._announcement_listing_limit("https://www.udom.ac.tz/blog/index")
    links = service._extract_recent_listing_links(
        soup,
        "https://www.udom.ac.tz/blog/index",
        limit,
        ["www.udom.ac.tz"],
        [],
    )

    assert limit == 5
    assert len(links) == 5
    assert links[-1] == ("https://www.udom.ac.tz/posts/5", "Blog 5")


def test_announcement_detail_pages_only_enqueue_pdf_links():
    service = CrawlerService.__new__(CrawlerService)
    soup = BeautifulSoup(
        """
        <a href="/files/notice.pdf">Notice PDF</a>
        <a href="/unrelated/page">Another page</a>
        <a href="https://example.com/file.pdf">External PDF</a>
        """,
        "html.parser",
    )

    links = service._extract_pdf_links(
        soup,
        "https://www.udom.ac.tz/item/1",
        ["www.udom.ac.tz"],
        [],
    )

    assert links == [("https://www.udom.ac.tz/files/notice.pdf", "Notice PDF")]


def test_crawler_detects_soft_not_found_pages():
    service = CrawlerService.__new__(CrawlerService)

    assert service._is_not_found_page(
        "Page Not Found",
        "404 Page Not Found The page you requested could not be found.",
    )


def test_crawler_does_not_treat_normal_policy_page_as_not_found():
    service = CrawlerService.__new__(CrawlerService)

    assert not service._is_not_found_page(
        "Examination Regulations",
        "Students must follow examination regulations. Missing documents are not found in this section only when not submitted.",
    )


def test_full_crawler_skips_announcement_blog_and_pdf_urls():
    service = CrawlerService.__new__(CrawlerService)

    assert service._should_skip_full_crawler_url("https://www.udom.ac.tz/announcements")
    assert service._should_skip_full_crawler_url("https://www.udom.ac.tz/blog/index")
    assert service._should_skip_full_crawler_url("https://storage.udom.ac.tz/public/file.pdf")


def test_full_crawler_skips_operational_subdomains_and_chrome_paths():
    service = CrawlerService.__new__(CrawlerService)

    assert service._should_skip_full_crawler_url("https://sr2.udom.ac.tz/")
    assert service._should_skip_full_crawler_url("https://portal.udom.ac.tz/")
    assert service._should_skip_full_crawler_url("https://www.udom.ac.tz/staff")
    assert service._should_skip_full_crawler_url("https://www.udom.ac.tz/gallery")
    assert not service._should_skip_full_crawler_url("https://www.udom.ac.tz/site/background")
