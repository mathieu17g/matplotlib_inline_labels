from matplotlib.text import Text

# TODO: ########################
# TODO: UNDER WORK             #
# TODO: ########################

class InlinedLabel(Text):
    """Defines an inlined label `Text` artist by storing its corresponing `Axes` information 
    on:
    - datalims
    - ...
    
    The `Text` draw method is reimplemented to detect conditions change affecting the 
    validity of the inlined label initially found best position and rotation
    """

    def __init__(self):
        super().__init__()
        self.initial_rotation = self.get_rotation()
        assert self.axes is not None
        self.initial_axes_data_ratio = self.axes.get_data_ratio()
        self.initial_axes_box_aspect = self.axes.get_box_aspect()
        self.initial_axes_xscale = self.axes.get_xscale()
        self.initial_axes_yscale = self.axes.get_yscale()

    def draw(self, renderer):
        self.update_rotation()
        super().draw(renderer)

    def update_rotation(self):
        pass