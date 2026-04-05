"""Unit tests for command routing."""


from cogniverse_messaging.command_router import parse_message


class TestCommandParsing:
    def test_plain_text_routes_to_routing_agent(self):
        result = parse_message(text="Show me videos of cats")
        assert result.agent_name == "routing_agent"
        assert result.query == "Show me videos of cats"
        assert not result.is_command

    def test_search_command(self):
        result = parse_message(text="/search machine learning tutorials")
        assert result.agent_name == "search_agent"
        assert result.query == "machine learning tutorials"
        assert result.is_command

    def test_summarize_command(self):
        result = parse_message(text="/summarize latest AI research")
        assert result.agent_name == "summarizer_agent"
        assert result.query == "latest AI research"

    def test_report_command(self):
        result = parse_message(text="/report quarterly trends")
        assert result.agent_name == "detailed_report_agent"
        assert result.query == "quarterly trends"

    def test_research_command(self):
        result = parse_message(text="/research deep learning architectures")
        assert result.agent_name == "deep_research_agent"
        assert result.query == "deep learning architectures"

    def test_code_command(self):
        result = parse_message(text="/code embedding generation")
        assert result.agent_name == "coding_agent"
        assert result.query == "embedding generation"

    def test_help_command(self):
        result = parse_message(text="/help")
        assert result.is_help
        assert result.is_command

    def test_start_with_token(self):
        result = parse_message(text="/start abc123def456")
        assert result.is_registration
        assert result.registration_token == "abc123def456"

    def test_start_without_token(self):
        result = parse_message(text="/start")
        assert result.is_registration
        assert result.registration_token is None

    def test_empty_command_shows_help(self):
        result = parse_message(text="/search")
        assert result.is_help

    def test_photo_routes_to_search(self):
        result = parse_message(
            has_photo=True, photo_file_id="AgACAgIAAxk"
        )
        assert result.agent_name == "search_agent"
        assert result.has_media
        assert result.media_type == "photo"
        assert result.media_file_id == "AgACAgIAAxk"

    def test_video_routes_to_search(self):
        result = parse_message(
            has_video=True, video_file_id="BAACAgIAAxk"
        )
        assert result.agent_name == "search_agent"
        assert result.has_media
        assert result.media_type == "video"

    def test_photo_with_caption(self):
        result = parse_message(
            text="Find similar scenes",
            has_photo=True,
            photo_file_id="AgACAgIAAxk",
        )
        assert result.agent_name == "search_agent"
        assert result.query == "Find similar scenes"
        assert result.has_media

    def test_empty_text(self):
        result = parse_message(text="")
        assert result.agent_name == "routing_agent"
        assert result.query == ""

    def test_none_text(self):
        result = parse_message(text=None)
        assert result.agent_name == "routing_agent"

    def test_wiki_save(self):
        result = parse_message(text="/wiki save")
        assert result.is_wiki
        assert result.is_command
        assert result.wiki_subcommand == "save"
        assert result.query == ""

    def test_wiki_search(self):
        result = parse_message(text="/wiki search machine learning")
        assert result.is_wiki
        assert result.is_command
        assert result.wiki_subcommand == "search"
        assert result.query == "machine learning"

    def test_wiki_topic(self):
        result = parse_message(text="/wiki topic transformers")
        assert result.is_wiki
        assert result.is_command
        assert result.wiki_subcommand == "topic"
        assert result.query == "transformers"

    def test_wiki_index(self):
        result = parse_message(text="/wiki index")
        assert result.is_wiki
        assert result.is_command
        assert result.wiki_subcommand == "index"
        assert result.query == ""
