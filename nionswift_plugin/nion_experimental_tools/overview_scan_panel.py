# standard libraries
import gettext
import typing

# nionswift libraries
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift import DocumentController
from nion.swift.model import PlugInManager
from nion.ui import Declarative
from nion.utils import Registry
from nion.typeshed import API_1_0


import time
import math
import numpy

from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller as stem_controller_module

import asyncio

_ = gettext.gettext


class SamplePanelUI:
    panel_type = "overview-scan-panel"

    def get_ui_handler(
            self,
            api_broker: PlugInManager.APIBroker,
            event_loop: typing.Optional[asyncio.AbstractEventLoop] = None,
            **kwargs: typing.Any,
    ) -> Declarative.HandlerLike:
        api = api_broker.get_api("~1.0")
        document_controller = kwargs.get("document_controller")
        return SamplePanelHandler(api, event_loop, document_controller)

class SamplePanelHandler(Declarative.Handler):
    """Declarative handler for the Sample docked panel."""

    def __init__(
            self,

            api: "API_1_0.API",
            event_loop: typing.Optional[asyncio.AbstractEventLoop],
            document_controller: typing.Any,
    ) -> None:
        super().__init__()
        self._api = api
        self._event_loop = event_loop or asyncio.get_event_loop()
        self._document_controller = document_controller
        self.width_value: str = ""
        self.height_value: str = ""
        self.defocus: str = ""
        self.instrument = typing.cast(stem_controller_module.STEMController, Registry.get_component('stem_controller'))
        self.camera = typing.cast(camera_base.CameraHardwareSource, self.instrument.ronchigram_camera)
        self._document_controller = document_controller
        self.output_text: str = ""
        self.progress_value: int = 0
        self.progress_max: int = 100
        self.progress_min: int = 0
        self.progress_text: str = "Progress:\nIdle"
        self._acq_task: typing.Optional[asyncio.Task[None]] = None
        self.ui_view = self._build_ui()

    def _set_progress(self, value: int, maximum: int, text: str) -> None:
        self.progress_value = value
        self.progress_max = max(1, int(maximum))
        self.progress_min = 0
        self.progress_text = text
        self.property_changed_event.fire("progress_value")
        self.property_changed_event.fire("progress_text")

    def _set_progress_threadsafe(self, value: int, maximum: int, text: str) -> None:
        self._event_loop.call_soon_threadsafe(self._set_progress, value, maximum, text)

    def _build_ui(self) -> typing.Mapping[str, typing.Any]:
        u = Declarative.DeclarativeUI()
        title = u.create_label(text="Overview Scan", font="bold")
        time_button = u.create_push_button(
            text="Estimate time to acquire survey image",
            on_clicked="on_estimate_time_clicked"
        )
        acq_button = u.create_push_button(
            text="Perform wide-field acquisition",
            on_clicked="on_perform_acquisition_clicked"
        )
        width_label = u.create_label(text="Desired width of image (um):")
        width_field = u.create_line_edit(text="@binding(width_value)", editable=True)

        height_label = u.create_label(text="Desired height of image (um):")
        height_field = u.create_line_edit(text="@binding(height_value)", editable=True)

        defocus_label = u.create_label(text="Desired defocus (nm):")
        defocus = u.create_line_edit(text="@binding(defocus)", editable=True)

        output_label = u.create_label(text="Output:")
        output_box = u.create_text_edit(
            text="@binding(output_text)",
            editable=False,
            height=200
        )
        progress_label = u.create_label(text="@binding(progress_text)")
        progress_bar = u.create_progress_bar(
            value="@binding(progress_value)",
            minimum=0,
            maximum=100,
            width=500
        )


        return u.create_column(
            title,
            u.create_row(width_label, u.create_spacing(4), width_field),
            u.create_spacing(4),
            u.create_row(height_label, u.create_spacing(4), height_field),
            u.create_spacing(4),
            u.create_row(defocus_label, u.create_spacing(4),  defocus),
            u.create_spacing(8),
            time_button,
            u.create_spacing(8),
            acq_button,
            u.create_spacing(8),
            output_label,
            output_box,
            u.create_spacing(50),
            progress_label,
            progress_bar,
            u.create_stretch(),
            width=500,
            height=500
        )

    def _append_output(self, message: str) -> None:
        self.output_text += f"{message}\n"
        self.property_changed_event.fire("output_text")

    def _append_output_threadsafe(self, message: str) -> None:
        self._event_loop.call_soon_threadsafe(self._append_output, message)

    def find_matrix(self, ds: float = 16e-6) -> numpy.ndarray:
        instrument = self.instrument
        sx0 = instrument.get_control_output("SShft.sx")
        sy0 = instrument.get_control_output("SShft.sy")
        x0 = instrument.get_control_output("SShft.x")
        y0 = instrument.get_control_output("SShft.y")

        instrument.set_control_output("SShft.sx", sx0 + ds)
        x1 = instrument.get_control_output("SShft.x")
        y1 = instrument.get_control_output("SShft.y")

        dx_from_sx = x1 - x0
        dy_from_sx = y1 - y0

        instrument.set_control_output("SShft.sx", sx0)
        instrument.set_control_output("SShft.sy", sy0)
        instrument.set_control_output("SShft.x", x0)
        instrument.set_control_output("SShft.y", y0)

        instrument.set_control_output("SShft.sy", sy0 + ds)
        x2 = instrument.get_control_output("SShft.x")
        y2 = instrument.get_control_output("SShft.y")
        instrument.set_control_output("SShft.sy", sy0)

        dx_from_sy = x2 - x0
        dy_from_sy = y2 - y0

        mat = numpy.array([
            [dx_from_sx / ds, dx_from_sy / ds],
            [dy_from_sx / ds, dy_from_sy / ds],
        ])

        return mat


    def acquisition(self,
                    instrument,
                    camera,
                    defocus,
                    target_width_m: tuple[float | int, float | int], timer = False,
                    reduce: float = 1.0):

        counter = 0

        try:
            tv_pixel_angle_rad = instrument.get_control_output("TVPixelAngle")
        except Exception:
            tv_pixel_angle_rad = None

        if tv_pixel_angle_rad is not None:
            shift_x_control_name = "SShft.sx"
            shift_y_control_name = "SShft.sy"

        else:
            shift_x_control_name = "stage_position_m.x"
            shift_y_control_name = "stage_position_m.y"

            frame = camera.grab_next_to_start()[0]
            assert frame is not None
            tv_pixel_angle_rad = frame.dimensional_calibrations[0].scale

        # grab stage original location
        sx_m = instrument.get_control_output(shift_x_control_name)
        sy_m = instrument.get_control_output(shift_y_control_name)
        df_original = instrument.get_control_output("C10")

        instrument.set_control_output("C10", defocus)
        pixel_size_m = abs(defocus) * math.tan(tv_pixel_angle_rad)

        image_size = camera.get_expected_dimensions(camera.get_current_frame_parameters())
        image_dtype = numpy.float32

        image_width_m = abs(defocus) * math.sin(tv_pixel_angle_rad * image_size[0])

        master_sub_area_size = image_size[0] // 2, image_size[1] // 2
        master_sub_area = (image_size[0] // 2 - master_sub_area_size[0] // 2, image_size[1] // 2 - master_sub_area_size[1] // 2), master_sub_area_size

        reduce = max(1, int(reduce))

        sub_area_shift_m = image_width_m * (master_sub_area[1][0] / image_size[0])

        sub_area = (master_sub_area[0][0] // reduce, master_sub_area[0][1] // reduce), (master_sub_area[1][0] // reduce, master_sub_area[1][1] // reduce)

        frames_needed_width = math.ceil(target_width_m[0] * 1e-6 / sub_area_shift_m)
        frames_needed_height = math.ceil(target_width_m[1] * 1e-6 / sub_area_shift_m)
        size = (frames_needed_width, frames_needed_height)

        total_images = frames_needed_width* frames_needed_height

        master_data = numpy.empty((sub_area[1][0] * size[0], sub_area[1][1] * size[1]), image_dtype)

        if not timer:
            self._append_output_threadsafe(f"Stage starting position: {sx_m * 1e6, sy_m * 1e6} um")
            self._append_output_threadsafe(f"Pixel size: {(pixel_size_m * 1e9):.3f} nm")
            self._append_output_threadsafe(f"Defocus: {(defocus * 1e9):.0f} nm")

            self._append_output_threadsafe(f"Image width: {image_width_m * 1e6} um")
            self._append_output_threadsafe(f"Master size: {master_data.shape}\n")

            self._set_progress_threadsafe(0, total_images, "Progress:\nStarting acquisition...")

        else:
            self._append_output_threadsafe(f"Need to acquire {frames_needed_width} x {frames_needed_height} frames for a {target_width_m[0]} um x {target_width_m[1]} um image \n")


        t1 = time.time()

        if timer:
            size = (1,1)
        else:
            size = size
        try:
            for row in range(size[0]):
                col_iter = range(size[1]) if (row % 2 == 0) else range(size[1] - 1, -1, -1)
                for column in col_iter:

                    if shift_x_control_name == "stage_position_m.x":
                        delta_x_m = - sub_area_shift_m * (column - size[1] // 2)
                        delta_y_m = - sub_area_shift_m * (row - size[0] // 2)
                    else:
                        matrix = self.find_matrix(instrument)
                        delta_x_m = - sub_area_shift_m * (column - size[1] // 2)
                        delta_y_m = - sub_area_shift_m * (row - size[0] // 2)
                        delta_camera = numpy.array([delta_x_m, delta_y_m], dtype=numpy.float64)
                        delta_fast = numpy.linalg.solve(matrix, delta_camera)

                        delta_x_m = float(delta_fast[0])
                        delta_y_m = float(delta_fast[1])

                    counter += 1

                    attempts = 0
                    while attempts < 4:
                        attempts += 1
                        try:
                            tolerance_factor = 0.0001
                            instrument.set_control_output(shift_x_control_name, sx_m - delta_x_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                            instrument.set_control_output(shift_y_control_name, sy_m - delta_y_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                        except TimeoutError:
                            self._append_output_threadsafe(f"Timeout row= {row} column= {column}")
                            continue
                        break

                    supradata = camera.grab_next_to_start()[0]
                    assert supradata is not None
                    # set both values
                    attempts = 0
                    while attempts < 4:
                        attempts += 1
                        try:
                            tolerance_factor = 0.0001
                            instrument.set_control_output(shift_x_control_name, sx_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                            instrument.set_control_output(shift_y_control_name, sy_m, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                        except TimeoutError:
                            self._append_output_threadsafe(f"Timeout row= {row} column= {column}")
                            continue
                        break
                    data = supradata.data[master_sub_area[0][0]:master_sub_area[0][0] + master_sub_area[1][0]:reduce, master_sub_area[0][1]:master_sub_area[0][1] + master_sub_area[1][1]:reduce]
                    slice_row = row
                    slice_column = column
                    slice0 = slice(slice_row * sub_area[1][0], (slice_row + 1) * sub_area[1][0])
                    slice1 = slice(slice_column * sub_area[1][1], (slice_column + 1) * sub_area[1][1])
                    master_data[slice0, slice1] = data

                    # inside loop after counter increment or frame write
                    if not timer:
                        pct = (100 * counter / total_images)
                        self._set_progress_threadsafe(pct, total_images, f"Progress:\nAcquiring {counter}/{total_images}")
            t2 = time.time()
            time_total = t2-t1
        finally:
            # restore stage to original location
            instrument.set_control_output(shift_x_control_name, sx_m)
            instrument.set_control_output(shift_y_control_name, sy_m)
            instrument.set_control_output("C10", df_original)

        if timer:
            return total_images, time_total
        else:
            return master_data


    def on_estimate_time_clicked(self, widget: typing.Any) -> None:
        try:
            width_um = int(self.width_value)
            height_um = int(self.height_value)
            defocus_nm = int(self.defocus) * 1e-9
        except ValueError:
            self._append_output("Please enter width, height, and defocus as integers.")
            return

        if width_um < 1 or height_um < 1:
            self._append_output("Please ensure width and height are positive.")
            return
        if width_um >= 1000 or height_um >= 1000:
            self._append_output("Warning: Requested scan size is outside of sensible limit")
            return
        if abs(defocus_nm * 1e9) < 1000 or abs(defocus_nm * 1e9) > 500000:
            self._append_output("Warning: Requested defocus is outside of sensible limit")
            return

        instrument = self.instrument
        camera = self.camera
        reduce = 1.0
        target_width_m = (width_um, height_um)
        total_images, t_total = self.acquisition(instrument, camera, defocus_nm, target_width_m, timer=True, reduce=reduce)

        time_taken = t_total * total_images
        self._append_output(
            f"This acquisition will take approximately {(time_taken // 3600):.0f}h {((time_taken % 3600) / 60):.0f}min {(time_taken % 60):.0f}s"
        )

    async def _run_acquisition_async(
        self,
        instrument,
        camera,
        defocus_nm: float,
        target_width_um: tuple[int, int],
    ) -> None:
        loop = self._event_loop

        try:
            master_data = await loop.run_in_executor(
                None,
                lambda: self.acquisition(instrument, camera, defocus_nm, target_width_um, False, 1.0),
            )
        except Exception as e:
            self._append_output(f"Acquisition failed: {e!r}")
            return

        try:
            library = self._api.library
            library.create_data_item_from_data(master_data, "Composite Survey")
            self._append_output("Acquisition complete.\n")
        except Exception as e:
            self._append_output(f"Failed to publish result: {e!r}")

    def on_perform_acquisition_clicked(self, widget: typing.Any) -> None:
        try:
            width_um = int(self.width_value)
            height_um = int(self.height_value)
            defocus_nm = int(self.defocus) * 1e-9
        except ValueError:
            self._append_output("Please enter width, height, and defocus as integers.")
            return

        if width_um <= 0 or height_um <= 0:
            self._append_output("Please ensure width and height are positive.")
            return

        if self._acq_task and not self._acq_task.done():
            self._append_output("Acquisition already running.")
            return

        instrument = self.instrument
        camera = self.camera
        target_width_um = (width_um, height_um)

        self._acq_task = self._event_loop.create_task(
            self._run_acquisition_async(instrument, camera, defocus_nm, target_width_um)
        )
# ---------------------------------------------------------------------------
# Swift Panel wrapper
# ---------------------------------------------------------------------------

class SamplePanel(Panel.Panel):
    """Swift panel class instantiated by the Workspace panel manager."""

    def __init__(
        self,
        document_controller: "DocumentController.DocumentController",
        panel_id: str,
        properties: typing.Dict[str, typing.Any],
    ) -> None:
        super().__init__(document_controller, panel_id, "overview-scan-panel")
        for component in Registry.get_components_by_type("overview-scan-panel"):
            if getattr(component, "panel_type", None) == "overview-scan-panel":
                ui_handler = component.get_ui_handler(
                    api_broker=PlugInManager.APIBroker(),
                    event_loop=document_controller.event_loop,
                    document_controller=document_controller,
                )
                self.widget = Declarative.DeclarativeWidget(
                    document_controller.ui,
                    document_controller.event_loop,
                    ui_handler,
                )
                break


class PanelSampleExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "sample.panel"

    def __init__(self, api_broker):
        self.__component = Registry.register_component(SamplePanelUI(), {"overview-scan-panel"})
        self.__panel = Workspace.WorkspaceManager().register_panel(
            SamplePanel,
            "sample-main-panel",
            _("Overview Scan"),
            ["left", "right"],
            "right",
            {"panel_type": "overview-scan-panel"},
        )

    def close(self):
        pass


# class SampleMenuItemDelegate:
#
#     def __init__(self, api):
#         self.__api = api
#         self.menu_id = "example_menu"  # required, specify menu_id where this item will go
#         self.menu_name = _("Examples")  # optional, specify default name if not a standard menu
#         self.menu_before_id = "window_menu"  # optional, specify before menu_id if not a standard menu
#         self.menu_item_name = _("Run Sample")  # menu item name
#
#     def menu_item_execute(self, window):
#         sampler.sample_function()
#
#
# class MenuSampleExtension:
#
#     # required for Swift to recognize this as an extension class.
#     extension_id = "sample.menu_item_call_sample"
#
#     def __init__(self, api_broker):
#         # grab the api object.
#         api = api_broker.get_api(version="~1.0")
#         # be sure to keep a reference or it will be closed immediately.
#         self.__menu_item_ref = api.create_menu_item(SampleMenuItemDelegate(api))
#
#     def close(self):
#         self.__menu_item_ref.close()
#         self.__menu_item_ref = None
