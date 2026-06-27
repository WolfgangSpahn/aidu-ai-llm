from aidu.ai.core.applet_info import AppletInfo


def test_applet_info_summarizes_selected_infostore_keys():
    applet_info = AppletInfo.from_payload(
        {
            "applet": "applet-periodic-table",
            "infoStore": {
                "elementName": "Lithium",
                "elementSymbol": "Li",
                "atomicNumber": 3,
                "unused": "hidden",
            },
        }
    )

    assert applet_info.to_text(("elementName", "atomicNumber")) == (
        "Applet event: applet-periodic-table with elementName=Lithium, atomicNumber=3"
    )


def test_applet_info_reads_structured_message_payload():
    applet_info = AppletInfo.from_message(
        {
            "role": "user",
            "kind": "applet",
            "content": "Applet event: applet-periodic-table",
            "applet_input": {
                "applet": "applet-periodic-table",
                "infoStore": {"elementSymbol": "Li"},
            },
        }
    )

    assert applet_info is not None
    assert applet_info.applet == "applet-periodic-table"
    assert applet_info.info_store == {"elementSymbol": "Li"}
