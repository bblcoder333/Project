import chromadb
from chromadb.utils import embedding_functions


def setup_knowledge_base():
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name="all-MiniLM-L6-v2"
             )
    chroma = chromadb.PersistentClient(path="./rag_knowledge_base")
    kb     = chroma.get_or_create_collection(
                 "ui_testing_knowledge",
                 embedding_function=ef
             )

    seed_knowledge = [

        # ── bare_minimum (minimal UI, single action screens) ──────────────
        ("tc_bare_min_primary_action",
         "Test case: On a bare minimum screen with a single primary action button, tap the button. Expected: the expected action is triggered and the user is navigated to the next screen.",
         {"type": "test_case", "module": "bare_minimum"}),

        ("ec_bare_min_double_tap",
         "Edge case: On a bare minimum screen, double tap the primary action button rapidly. Expected: action triggers only once, no duplicate submissions or crashes.",
         {"type": "edge_case", "module": "bare_minimum"}),

        # ── camera (camera viewfinder, capture screens) ───────────────────
        ("tc_camera_capture",
         "Test case: On a camera screen, tap the capture button. Expected: photo or video is captured and saved, confirmation feedback is shown.",
         {"type": "test_case", "module": "camera"}),

        ("ec_camera_permissions",
         "Edge case: On a camera screen, deny camera permissions. Expected: a permission explanation is shown and the app does not crash.",
         {"type": "edge_case", "module": "camera"}),

        # ── chat (messaging, conversation screens) ────────────────────────
        ("tc_chat_send_message",
         "Test case: On a chat screen, type a message in the input field and tap send. Expected: message appears in the conversation thread immediately.",
         {"type": "test_case", "module": "chat"}),

        ("ec_chat_empty_message",
         "Edge case: On a chat screen, tap the send button without typing anything. Expected: no empty message is sent, send button is disabled or ignored.",
         {"type": "edge_case", "module": "chat"}),

        ("mr_chat_whitespace",
         "Metamorphic relation: Sending a message with leading or trailing whitespace should produce the same delivered result as the trimmed message.",
         {"type": "metamorphic", "module": "chat"}),

        # ── content (article, text content, reader screens) ───────────────
        ("tc_content_scroll",
         "Test case: On a content screen, scroll through the full article or text. Expected: content loads progressively, no blank sections or crashes.",
         {"type": "test_case", "module": "content"}),

        ("tc_content_share",
         "Test case: On a content screen with a share button, tap share. Expected: native share sheet appears with available sharing options.",
         {"type": "test_case", "module": "content"}),

        # ── dashboard (home, summary, overview screens) ───────────────────
        ("tc_dashboard_load",
         "Test case: Navigate to the dashboard screen. Expected: all summary widgets and data sections load within 3 seconds with no empty states.",
         {"type": "test_case", "module": "dashboard"}),

        ("tc_dashboard_refresh",
         "Test case: On a dashboard screen, perform a pull-to-refresh gesture. Expected: data refreshes and updated content is shown.",
         {"type": "test_case", "module": "dashboard"}),

        # ── date_picker (calendar, date/time selection screens) ───────────
        ("tc_date_picker_select",
         "Test case: On a date picker screen, select a future date and confirm. Expected: selected date is returned to the calling screen and displayed correctly.",
         {"type": "test_case", "module": "date_picker"}),

        ("ec_date_picker_past",
         "Edge case: On a date picker screen that requires a future date, attempt to select a past date. Expected: past dates are disabled or an error is shown.",
         {"type": "edge_case", "module": "date_picker"}),

        # ── empty_state (no data, zero results screens) ───────────────────
        ("tc_empty_state_cta",
         "Test case: On an empty state screen, tap the call-to-action button. Expected: user is navigated to the relevant creation or onboarding flow.",
         {"type": "test_case", "module": "empty_state"}),

        # ── form (data entry, multi-field input screens) ──────────────────
        ("tc_form_submit_valid",
         "Test case: On a form screen, fill all required fields with valid data and submit. Expected: form is submitted successfully and user sees a confirmation.",
         {"type": "test_case", "module": "form"}),

        ("tc_form_submit_empty",
         "Test case: On a form screen, leave required fields empty and tap submit. Expected: validation errors are shown next to each required field, form is not submitted.",
         {"type": "test_case", "module": "form"}),

        ("ec_form_long_input",
         "Edge case: On a form screen, enter 500 or more characters in a text input field. Expected: input is gracefully truncated or a character limit message is shown.",
         {"type": "edge_case", "module": "form"}),

        ("ec_form_special_chars",
         "Edge case: On a form screen, enter special characters such as angle brackets and SQL quotes in a text field. Expected: input is sanitized, no crash or server error.",
         {"type": "edge_case", "module": "form"}),

        ("mr_form_whitespace",
         "Metamorphic relation: Submitting a form with leading or trailing whitespace in text fields should produce the same result as submitting the trimmed values.",
         {"type": "metamorphic", "module": "form"}),

        # ── gallery (image grid, media collection screens) ────────────────
        ("tc_gallery_scroll",
         "Test case: On a gallery screen, scroll through the image grid. Expected: images load progressively, no blank tiles, no crashes.",
         {"type": "test_case", "module": "gallery"}),

        ("tc_gallery_open_item",
         "Test case: On a gallery screen, tap an image thumbnail. Expected: full-size image viewer opens with the correct image.",
         {"type": "test_case", "module": "gallery"}),

        # ── list (scrollable list, feed, catalog screens) ─────────────────
        ("tc_list_scroll",
         "Test case: On a list screen, scroll to the bottom of the list. Expected: all items load correctly and a end-of-list indicator or pagination appears.",
         {"type": "test_case", "module": "list"}),

        ("tc_list_item_tap",
         "Test case: On a list screen, tap a list item. Expected: detail screen for that item opens with correct information.",
         {"type": "test_case", "module": "list"}),

        # ── login (authentication, sign in screens) ───────────────────────
        ("tc_login_valid",
         "Test case: On a login screen, enter a valid username and password and tap Log In. Expected: user is authenticated and navigated to the home or dashboard screen.",
         {"type": "test_case", "module": "login"}),

        ("tc_login_empty",
         "Test case: On a login screen, leave username and password empty and tap Log In. Expected: validation error is shown, user is not authenticated.",
         {"type": "test_case", "module": "login"}),

        ("tc_login_wrong_password",
         "Test case: On a login screen, enter a valid username with an incorrect password. Expected: error message is shown, user remains on the login screen.",
         {"type": "test_case", "module": "login"}),

        ("ec_login_sql_injection",
         "Edge case: On a login screen, enter a SQL injection string in the username field. Expected: input is sanitized, no database error or unauthorized access.",
         {"type": "edge_case", "module": "login"}),

        ("mr_login_case",
         "Metamorphic relation: For case-insensitive username fields, uppercase and lowercase versions of the same username should produce the same authentication result.",
         {"type": "metamorphic", "module": "login"}),

        # ── map (location, navigation, map view screens) ──────────────────
        ("tc_map_load",
         "Test case: On a map screen, allow location permissions. Expected: map loads centered on the user's current location with correct pins or overlays.",
         {"type": "test_case", "module": "map"}),

        ("ec_map_no_location",
         "Edge case: On a map screen, deny location permissions. Expected: a fallback default location or permission explanation is shown, app does not crash.",
         {"type": "edge_case", "module": "map"}),

        # ── menu (navigation drawer, sidebar, hamburger menu screens) ─────
        ("tc_menu_open_close",
         "Test case: On a screen with a hamburger menu, tap the menu icon to open it then tap outside to close. Expected: menu opens and closes smoothly with no layout issues.",
         {"type": "test_case", "module": "menu"}),

        ("tc_menu_navigate",
         "Test case: On a menu screen, tap each navigation item. Expected: each item navigates to the correct screen.",
         {"type": "test_case", "module": "menu"}),

        # ── modal (dialog, popup, bottom sheet screens) ───────────────────
        ("tc_modal_dismiss",
         "Test case: On a modal dialog screen, tap outside the modal or the dismiss button. Expected: modal closes and the underlying screen is restored.",
         {"type": "test_case", "module": "modal"}),

        ("tc_modal_confirm",
         "Test case: On a modal dialog with confirm and cancel options, tap confirm. Expected: the intended action is executed and the modal closes.",
         {"type": "test_case", "module": "modal"}),

        # ── onboarding (walkthrough, tutorial, intro screens) ─────────────
        ("tc_onboarding_next",
         "Test case: On an onboarding screen, tap the next button to advance through all steps. Expected: each step advances correctly and the final step leads to the main app.",
         {"type": "test_case", "module": "onboarding"}),

        ("tc_onboarding_skip",
         "Test case: On an onboarding screen with a skip option, tap skip. Expected: user is taken directly to the main app or registration screen.",
         {"type": "test_case", "module": "onboarding"}),

        # ── profile (user profile, account screens) ───────────────────────
        ("tc_profile_view",
         "Test case: Navigate to a profile screen. Expected: all user information including name, avatar, and stats loads correctly.",
         {"type": "test_case", "module": "profile"}),

        ("tc_profile_edit",
         "Test case: On a profile screen, tap edit, change the display name, and save. Expected: updated name is shown on the profile screen after saving.",
         {"type": "test_case", "module": "profile"}),

        # ── search (search input, results screens) ────────────────────────
        ("tc_search_valid",
         "Test case: On a search screen, enter a valid search term and submit. Expected: relevant results are displayed within 3 seconds.",
         {"type": "test_case", "module": "search"}),

        ("tc_search_empty",
         "Test case: On a search screen, submit an empty search query. Expected: helpful prompt or recent searches shown, no crash.",
         {"type": "test_case", "module": "search"}),

        ("ec_search_special_chars",
         "Edge case: On a search screen, enter special characters in the search field. Expected: results or empty state shown, no crash or server error.",
         {"type": "edge_case", "module": "search"}),

        # ── settings (preferences, configuration screens) ─────────────────
        ("tc_settings_toggle",
         "Test case: On a settings screen, tap a toggle switch to change its state. Expected: toggle visually changes and the setting is persisted on next app open.",
         {"type": "test_case", "module": "settings"}),

        ("tc_settings_back",
         "Test case: On a settings screen, tap the back button. Expected: user is returned to the previous screen with no data loss.",
         {"type": "test_case", "module": "settings"}),

        # ── terms (terms of service, privacy policy screens) ──────────────
        ("tc_terms_scroll",
         "Test case: On a terms of service screen, scroll through the full content. Expected: entire text is readable with no clipping or overlap.",
         {"type": "test_case", "module": "terms"}),

        ("tc_terms_accept",
         "Test case: On a terms screen with an accept button, tap accept. Expected: user proceeds to the next step and agreement is recorded.",
         {"type": "test_case", "module": "terms"}),

        # ── general metamorphic relations (apply to all screen types) ─────
        ("mr_orientation",
         "Metamorphic relation: Rotating the device from portrait to landscape should display the same content and functionality with an adapted layout.",
         {"type": "metamorphic", "module": "general"}),

        ("mr_back_navigation",
         "Metamorphic relation: Pressing the back button from any screen should return the user to the previous screen with the same state as when they left it.",
         {"type": "metamorphic", "module": "general"}),

        ("mr_interruption",
         "Metamorphic relation: Receiving a phone call or notification during any interaction and then resuming should restore the app to the exact same state.",
         {"type": "metamorphic", "module": "general"}),
    ]

    if kb.count() == 0 and seed_knowledge:
        kb.add(
            ids       = [s[0] for s in seed_knowledge],
            documents = [s[1] for s in seed_knowledge],
            metadatas = [s[2] for s in seed_knowledge],
        )
        print(f"✅ Knowledge base populated — {kb.count()} documents")
    else:
        print(f"✅ Knowledge base ready — {kb.count()} documents")

    return kb