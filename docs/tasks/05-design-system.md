Task 05 — GUI Design System and Reusable Widgets
1. Objective
Create the visual foundation for a modern, consistent scientific desktop application using PySide6.
This task establishes shared styles and reusable widgets before implementing the complete application pages.
2. Prerequisites
- Task 00 is complete.- docs/ui-design-system.md is available.- The GUI framework and supported Python version are documented.
3. Scope
Implement:
- Centralized colors and typography.- Application stylesheet.- Shared visual constants.- Reusable buttons, labels, cards, and status indicators where needed.- Empty-state component.- Error and warning presentation patterns.- Shared spacing and layout conventions.- Basic accessibility behavior.
Do not implement the full dataset, training, or recall workflows in this task.
4. Expected Modules
- src/associative_memory/presentation/theme/palette.py- src/associative_memory/presentation/theme/typography.py- src/associative_memory/presentation/theme/styles.qss- src/associative_memory/presentation/theme/icons.py- src/associative_memory/presentation/widgets/metric_card.py- src/associative_memory/presentation/widgets/empty_state.py- Additional shared widgets only when justified.- Relevant GUI tests.
5. Visual Direction
The application should look like a polished scientific workstation.
Use:
- A calm neutral background.- White or neutral content surfaces.- A restrained indigo or blue primary accent.- Green for confirmed success.- Amber for warnings.- Red for errors.- Consistent borders, radii, spacing, and typography.- A monospaced font for vectors, matrices, and numerical traces.
Avoid:
- Excessive gradients.- Strong shadows on every component.- Unnecessary animations.- Randomly chosen colors per screen.- Overly dense layouts.- Emoji as replacements for essential interface icons.
6. Reusable Component Requirements
Metric Card
Must support:
- Title.- Value.- Optional description.- Optional status.- Sensible behavior when a value is unavailable.
Empty State
Must support:
- Clear explanation of why content is absent.- Optional next-action button.- Consistent spacing and typography.
Status Presentation
Must distinguish:
- Success.- Information.- Warning.- Error.- In-progress state.
Use text or icons in addition to color.
Icon System
Use a consistent icon source and size policy.
Ensure icons are available in the packaged application. Avoid relying on arbitrary local paths or external network downloads at runtime.
7. Accessibility and Internationalization
- Support readable contrast.- Provide accessible labels for interactive controls.- Provide keyboard focus indicators.- Avoid relying exclusively on hover interactions.- Ensure technical values remain readable at normal window sizes.- Keep a documented strategy for Arabic and English text.- Do not manually reverse mathematical expressions or numerical vectors to imitate right-to-left rendering.
8. Theme Architecture
Centralize reusable styles.
Do not repeat long style strings in every view.
Keep domain and algorithm modules entirely independent of GUI styling.
Do not add a theme-switching system unless it can be implemented without complicating the initial design; dark mode can remain a later enhancement.
9. Required Tests
- Shared widgets can be instantiated.- Metric cards handle missing values.- Empty states render without requiring application data.- Stylesheet loading works from the intended application entry point.- Interactive controls have accessible labels or appropriate accessible names.- Tests run without requiring a real display where headless testing is supported.
10. Acceptance Criteria
- The theme is centralized.- Shared widgets can be reused by all subsequent pages.- Typography and spacing are consistent.- Empty and error states have deliberate designs.- No business logic is added to presentation-only widgets.- The application can display a minimal styled window.
11. Completion Report
Report the theme files, reusable widgets, visual decisions, tests executed, and any remaining accessibility limitations.
Do not begin Task 06 until the basic design system loads successfully.