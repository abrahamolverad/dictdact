"""The menus a regular (Dock) app is expected to have — including Edit, without which
⌘C / ⌘V do nothing inside the dashboard's web view."""
from __future__ import annotations

import objc
from AppKit import NSApp, NSMenu, NSMenuItem
from Foundation import NSObject

NSCommandKeyMask, NSAlternateKeyMask = 1 << 20, 1 << 19


def _item(menu, title, action, key, target=None, modifiers=None):
    it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
    if target is not None:
        it.setTarget_(target)
    if modifiers is not None:
        it.setKeyEquivalentModifierMask_(modifiers)
    menu.addItem_(it)
    return it


def build_main_menu(controller) -> NSMenu:
    """``controller`` must respond to showWindow:, showSettings: and quitDictDact:."""
    main = NSMenu.alloc().init()

    app_item = NSMenuItem.alloc().init()
    main.addItem_(app_item)
    app_menu = NSMenu.alloc().initWithTitle_("DictDact")
    _item(app_menu, "Open DictDact", "showWindow:", "0", controller)
    _item(app_menu, "Settings…", "showSettings:", ",", controller)
    app_menu.addItem_(NSMenuItem.separatorItem())
    _item(app_menu, "Hide DictDact", "hide:", "h")
    _item(app_menu, "Hide Others", "hideOtherApplications:", "h",
          modifiers=NSCommandKeyMask | NSAlternateKeyMask)
    _item(app_menu, "Show All", "unhideAllApplications:", "")
    app_menu.addItem_(NSMenuItem.separatorItem())
    _item(app_menu, "Quit DictDact", "quitDictDact:", "q", controller)
    app_item.setSubmenu_(app_menu)

    edit_item = NSMenuItem.alloc().init()
    main.addItem_(edit_item)
    edit_menu = NSMenu.alloc().initWithTitle_("Edit")
    _item(edit_menu, "Undo", "undo:", "z")
    _item(edit_menu, "Redo", "redo:", "Z")
    edit_menu.addItem_(NSMenuItem.separatorItem())
    _item(edit_menu, "Cut", "cut:", "x")
    _item(edit_menu, "Copy", "copy:", "c")
    _item(edit_menu, "Paste", "paste:", "v")
    _item(edit_menu, "Select All", "selectAll:", "a")
    edit_item.setSubmenu_(edit_menu)

    win_item = NSMenuItem.alloc().init()
    main.addItem_(win_item)
    win_menu = NSMenu.alloc().initWithTitle_("Window")
    _item(win_menu, "Minimize", "performMiniaturize:", "m")
    _item(win_menu, "Close", "performClose:", "w")
    win_item.setSubmenu_(win_menu)
    NSApp.setWindowsMenu_(win_menu)
    return main


class MenuController(NSObject):
    """Bridges the AppKit menu items to the Python app (rumps.App is not an NSObject)."""

    def initWithApp_(self, app):
        self = objc.super(MenuController, self).init()
        if self is None:
            return None
        self._app = app
        return self

    def showWindow_(self, sender):
        self._app._show_window()

    def showSettings_(self, sender):
        self._app._show_window("&tab=settings")

    def quitDictDact_(self, sender):
        self._app._quit()
