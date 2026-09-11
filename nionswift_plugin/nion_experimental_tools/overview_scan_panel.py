import typing

import asyncio
import gettext
import math
import numpy
import numpy.typing as npt
from pathlib import Path
from PIL import Image
import time

from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller as stem_controller_module
from nion.swift import DocumentController
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import PlugInManager
from nion.typeshed import API_1_0
from nion.ui import Declarative
from nion.utils import Model
from nion.utils import Registry

_ = gettext.gettext
JSONDict = dict[str, typing.Any]

class OverviewScanPanelUI:
    panel_type = "overview-scan-panel"

    @staticmethod
    def get_ui_handler(
            api_broker: PlugInManager.APIBroker,
            event_loop: typing.Optional[asyncio.AbstractEventLoop] = None,
            **kwargs: typing.Any,
    ) -> Declarative.HandlerLike:
        api = api_broker.get_api("~1.0")
        document_controller = kwargs.get("document_controller")
        return OverviewSamplePanelHandler(api, event_loop, document_controller)


class OverviewSamplePanelHandler(Declarative.Handler):

    def __init__(
            self,

            api: "API_1_0.API",
            event_loop: typing.Optional[asyncio.AbstractEventLoop],
            document_controller: typing.Any,
    ) -> None:
        super().__init__()
        self._api = api
        self._event_loop = event_loop or asyncio.get_event_loop()
        self.stem_controller = typing.cast(stem_controller_module.STEMController, Registry.get_component('stem_controller'))
        self.camera = typing.cast(camera_base.CameraHardwareSource, self.stem_controller.ronchigram_camera)
        self._document_controller = document_controller
        self.output_text: str = ""
        self.progress_value: int = 0
        self.progress_max: int = 100
        self.progress_min: int = 0
        self.progress_text: str = "Progress:\nIdle"
        self._acq_task: typing.Optional[asyncio.Task[None]] = None
        self.width_value: str = "30"
        self.height_value: str = "30"
        self.defocus: str = "-50000"
        self.binning: str = "1"
        self._cancel_requested: bool = False
        self._is_running: bool = False
        self.cancel_enabled = Model.PropertyModel(False)
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

    @staticmethod
    def _build_ui() -> typing.Mapping[str, typing.Any]:
        u = Declarative.DeclarativeUI()
        title = u.create_label(text="Overview Scan", font="bold")
        time_button = u.create_push_button(text="Estimate scan size and duration", on_clicked="handle_estimate_time_clicked")
        acq_button = u.create_push_button(text="Scan", on_clicked="handle_perform_acquisition_clicked")
        properties_label = u.create_label(text="Desired properties of image:")
        width_label = u.create_label(text="Width (um):")
        width_field = u.create_line_edit(text="@binding(width_value)", editable=True)
        height_label = u.create_label(text="Height (um):")
        height_field = u.create_line_edit(text="@binding(height_value)", editable=True)
        defocus_label = u.create_label(text="Defocus (nm):")
        defocus = u.create_line_edit(text="@binding(defocus)", editable=True)
        reduce_label = u.create_label(text="Binning:")
        reduce_val = u.create_line_edit(text="@binding(binning)", editable=True)
        output_label = u.create_label(text="Output:")
        output_box = u.create_text_edit(text="@binding(output_text)", editable=False, height=200)
        progress_label = u.create_label(text="@binding(progress_text)")
        progress_bar = u.create_progress_bar(value="@binding(progress_value)", minimum=0, maximum=100, width=500)
        cancel_button = u.create_push_button(text="Cancel", on_clicked="handle_cancel_acquisition_clicked", enabled="@binding(cancel_enabled.value)")
        clear_button = u.create_push_button(text="Clear minimap", on_clicked="handle_clear_minimap_clicked")

        return typing.cast(typing.Mapping[str, typing.Any], u.create_column(
            title,
            properties_label,
            u.create_row(width_label, u.create_spacing(4), width_field, u.create_spacing(20), height_label, u.create_spacing(4), height_field),
            u.create_spacing(4),
            u.create_row(defocus_label, u.create_spacing(4), defocus, u.create_spacing(20), reduce_label, u.create_spacing(4), reduce_val),
            u.create_spacing(8),
            u.create_row(time_button, u.create_spacing(20), acq_button),
            u.create_spacing(8),
            progress_label,
            progress_bar,
            u.create_spacing(8),
            u.create_row(cancel_button, u.create_spacing(4),clear_button),
            u.create_spacing(8),
            output_label,
            output_box,
            u.create_stretch(),
            margin=6,
            spacing=4
        ))

    def _append_output(self, message: str) -> None:
        self.output_text += f"{message}\n"
        self.property_changed_event.fire("output_text")

    def _append_output_threadsafe(self, message: str) -> None:
        self._event_loop.call_soon_threadsafe(self._append_output, message)

    def handle_cancel_acquisition_clicked(self, widget: typing.Any) -> None:
        if self._is_running:
            self._cancel_requested = True
            self._set_progress_threadsafe(self.progress_value, 100, "Cancel requested...")

    def find_matrix(self, ds: float = 16e-6) -> numpy.ndarray:
        stem_controller = self.stem_controller
        sx0 = stem_controller.get_control_output("SShft.sx")
        sy0 = stem_controller.get_control_output("SShft.sy")
        x0 = stem_controller.get_control_output("SShft.x")
        y0 = stem_controller.get_control_output("SShft.y")

        stem_controller.set_control_output("SShft.sx", sx0 + ds)
        x1 = stem_controller.get_control_output("SShft.x")
        y1 = stem_controller.get_control_output("SShft.y")

        dx_from_sx = x1 - x0
        dy_from_sx = y1 - y0

        stem_controller.set_control_output("SShft.sx", sx0)
        stem_controller.set_control_output("SShft.sy", sy0)
        stem_controller.set_control_output("SShft.x", x0)
        stem_controller.set_control_output("SShft.y", y0)

        stem_controller.set_control_output("SShft.sy", sy0 + ds)
        x2 = stem_controller.get_control_output("SShft.x")
        y2 = stem_controller.get_control_output("SShft.y")
        stem_controller.set_control_output("SShft.sy", sy0)

        dx_from_sy = x2 - x0
        dy_from_sy = y2 - y0

        mat = numpy.array([
            [dx_from_sx / ds, dx_from_sy / ds],
            [dy_from_sx / ds, dy_from_sy / ds],
        ])

        return mat

    def acquisition(self,
                    stem_controller: stem_controller_module.STEMController,
                    camera: camera_base.CameraHardwareSource,
                    defocus: float,
                    target_width_um: tuple[float | int, float | int], timer: bool = False,
                    reduce: float = 1.0) -> (tuple[npt.NDArray[numpy.float64], int, float] |
                                             tuple[npt.NDArray[numpy.float64], tuple[tuple[int, int], tuple[int, int]], float, float, float, float, float] |
                                             tuple[int, float] | None):
        counter = 0
        self._cancel_requested = False
        self._is_running = True
        self.cancel_enabled.value = True

        success, tv_pixel_angle_rad = stem_controller.TryGetVal("TVPixelAngle")

        if success:
            shift_x_control_name = "SShft.sx"
            shift_y_control_name = "SShft.sy"
            matrix = self.find_matrix()

        else:
            shift_x_control_name = "stage_position_m.x"
            shift_y_control_name = "stage_position_m.y"
            matrix = None

            frame = camera.grab_next_to_start()[0]
            assert frame is not None
            tv_pixel_angle_rad = float(frame.dimensional_calibrations[0].scale)

        # grab stage original location and original defocus
        sx_um = stem_controller.get_control_output(shift_x_control_name)
        sy_um = stem_controller.get_control_output(shift_y_control_name)
        df_original = stem_controller.get_control_output("C10")

        assert tv_pixel_angle_rad is not None
        stem_controller.set_control_output("C10", defocus)

        pixel_size_nm = abs(defocus) * math.tan(tv_pixel_angle_rad)
        image_size = camera.get_expected_dimensions(camera.get_current_frame_parameters())
        image_width_um = abs(defocus) * math.sin(tv_pixel_angle_rad * image_size[0])

        master_sub_area_size = image_size[0], image_size[1]
        master_sub_area = (image_size[0] // 2 - master_sub_area_size[0] // 2, image_size[1] // 2 - master_sub_area_size[1] // 2), master_sub_area_size
        reduce = max(1, int(reduce))

        sub_area_shift_um = image_width_um * (master_sub_area[1][0] / image_size[0])
        sub_area = (master_sub_area[0][0] // reduce, master_sub_area[0][1] // reduce), (master_sub_area[1][0] // reduce, master_sub_area[1][1] // reduce)

        frames_needed_width = math.ceil(target_width_um[0] * 1e-6 / sub_area_shift_um)
        frames_needed_height = math.ceil(target_width_um[1] * 1e-6 / sub_area_shift_um)
        size = (frames_needed_width, frames_needed_height)
        total_image_height = size[1] * image_width_um
        total_images = frames_needed_width * frames_needed_height

        master_data = numpy.empty((sub_area[1][0] * size[0], sub_area[1][1] * size[1]))

        if not timer:
            self._append_output_threadsafe(f"Stage starting position: {sx_um * 1e6, sy_um * 1e6} um")
            self._append_output_threadsafe(f"Pixel size: {(pixel_size_nm * 1e9):.3f} nm")
            self._append_output_threadsafe(f"Defocus: {(defocus * 1e9):.0f} nm")

            self._append_output_threadsafe(f"Frame width: {image_width_um * 1e6} um")
            self._append_output_threadsafe(f"Master size: {master_data.shape}\n")

            self._set_progress_threadsafe(0, total_images, "Progress:\nStarting acquisition...")

        t1 = time.time()

        if timer:
            size = (2, 1)
        else:
            size = size

        try:
            for row in range(size[0]):
                if self._cancel_requested:
                    self._append_output_threadsafe("Acquisition Cancelled.")
                    self.cancel_enabled.value = False
                    return None if not timer else (0, 0.0)

                col_iter = range(size[1]) if (row % 2 == 0) else range(size[1] - 1, -1, -1)
                for column in col_iter:
                    if self._cancel_requested:
                        self._append_output_threadsafe("Acquisition Cancelled.")
                        self.cancel_enabled.value = False
                        return None if not timer else (0, 0.0)

                    if matrix is None or numpy.linalg.det(matrix) == 0 or len(matrix) == 0:
                        delta_x_um = - sub_area_shift_um * (column - size[1] // 2)
                        delta_y_um = - sub_area_shift_um * (row - size[0] // 2)
                    else:
                        delta_x_um = - sub_area_shift_um * (column - size[1] // 2)
                        delta_y_um = - sub_area_shift_um * (row - size[0] // 2)
                        delta_camera = numpy.array([delta_x_um, delta_y_um], dtype=numpy.float64)
                        delta_fast = numpy.linalg.solve(matrix, delta_camera)

                        delta_x_um = float(delta_fast[0])
                        delta_y_um = float(delta_fast[1])

                    counter += 1

                    attempts = 0
                    while attempts < 4:
                        if self._cancel_requested:
                            self._append_output_threadsafe("Acquisition Cancelled.")
                            self.cancel_enabled.value = False
                            return None if not timer else (0, 0.0)
                        attempts += 1
                        try:
                            tolerance_factor = 0.0001
                            stem_controller.set_control_output(shift_x_control_name, sx_um - delta_x_um, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                            stem_controller.set_control_output(shift_y_control_name, sy_um - delta_y_um, {"confirm": True, "confirm_tolerance_factor": tolerance_factor})
                        except TimeoutError:
                            self._append_output_threadsafe(f"Timeout row= {row} column= {column}")
                            continue
                        break

                    supradata = camera.grab_next_to_start()[0]
                    assert supradata is not None
                    data = supradata.data[master_sub_area[0][0]:master_sub_area[0][0] + master_sub_area[1][0]:reduce, master_sub_area[0][1]:master_sub_area[0][1] + master_sub_area[1][1]:reduce]
                    slice_row = row
                    slice_column = column
                    slice0 = slice(slice_row * sub_area[1][0], (slice_row + 1) * sub_area[1][0])
                    slice1 = slice(slice_column * sub_area[1][1], (slice_column + 1) * sub_area[1][1])
                    master_data[slice0, slice1] = data

                    # inside loop after counter increment or frame write
                    if not timer:
                        pct = int(100 * counter / total_images)
                        self._set_progress_threadsafe(pct, total_images, f"Progress:\nAcquiring frame {counter} of {total_images}")
            t2 = time.time()
            time_total = t2 - t1
        finally:
            # restore stage to original location
            stem_controller.set_control_output(shift_x_control_name, sx_um)
            stem_controller.set_control_output(shift_y_control_name, sy_um)
            stem_controller.set_control_output("C10", df_original)
            self._set_progress_threadsafe(0, 100, "Progress:\n Idle")

        if timer:
            self.cancel_enabled.value = False
            return master_data, total_images, time_total
        else:
            self.cancel_enabled.value = False
            return master_data, sub_area, sub_area_shift_um, pixel_size_nm, total_image_height, sx_um, sy_um

    def handle_estimate_time_clicked(self, widget: typing.Any) -> None:
        try:
            width_um = int(self.width_value)
            height_um = int(self.height_value)
            defocus_nm = int(self.defocus) * 1e-9
            reduce = int(self.binning)
        except ValueError:
            self._append_output("Please enter width, height, binning and defocus as integers.")
            return
        if width_um < 1 or height_um < 1 or reduce < 1:
            self._append_output("Please ensure width and height are positive.")
            return
        if width_um >= 1000 or height_um >= 1000:
            self._append_output("Warning: Requested scan size is outside of sensible limit")
            return

        if abs(defocus_nm * 1e9) < 1000 or abs(defocus_nm * 1e9) > 500000:
            self._append_output("Warning: Requested defocus is outside of sensible limit")
            return
        stem_controller = self.stem_controller
        camera = self.camera

        target_width_um = (width_um, height_um)
        result = self.acquisition(stem_controller, camera, defocus_nm, target_width_um, timer=True, reduce=reduce)
        if result is None or len(result) != 3:
            return
        master_data, total_images, t_total = result
        image_size = master_data.shape
        time_taken = t_total * total_images / 2  # average time to move the stage
        self._append_output(
            f"This acquisition will take approximately {(time_taken // 3600):.0f}h {((time_taken % 3600) / 60):.0f}min {(time_taken % 60):.0f}s"
        )
        self._append_output(f"The size of the final data item will be {image_size}.\n")
        if any(dim > 32768 for dim in image_size):
            self._append_output("The final data item is too large to be used in the sample navigation window. Consider increasing the binning or reducing the size of the acquisition.\n")
            return
        else:
            return

    async def _run_acquisition_async(
        self,
        stem_controller: stem_controller_module.STEMController,
        camera: camera_base.CameraHardwareSource,
        defocus_nm: float,
        target_width_um: tuple[int, int],
        reduce: int
    ) -> None:
        loop = self._event_loop

        self._append_output_threadsafe("Starting acquisition...\n")
        try:
            result = await loop.run_in_executor(
                None, self.acquisition, stem_controller, camera, defocus_nm, target_width_um, False, reduce
            )
            if result is None or len(result) != 7:
                self._set_progress(0, 100, "Progress:\nIdle")
                return

            master_data, sub_area, sub_area_shift_m, pixel_size_m, total_image_height, sx_um, sy_um = result
        except Exception as e:
            self._append_output(f"Acquisition failed: {e!r}")
            self.cancel_enabled.value = False
            return

        try:
            library = self._api.library
            y_scale_um = (sub_area_shift_m / sub_area[1][0]) * 1e6
            x_scale_um = (sub_area_shift_m / sub_area[1][1]) * 1e6
            dimensional_calibrations = [
                self._api.create_calibration(0.0, y_scale_um, "um"),
                self._api.create_calibration(0.0, x_scale_um, "um"),
            ]

            xdata = self._api.create_data_and_metadata(
                master_data,
                dimensional_calibrations=dimensional_calibrations,
            )

            library.create_data_item_from_data_and_metadata(xdata, "Composite Survey")
            self._append_output("Acquisition complete.\n")

            self._append_output("Image properties:")
            self._append_output_threadsafe(f"Total image height: {total_image_height * 1e3} mm")
            self._append_output_threadsafe(f"Original stage coordinates: {sx_um * 1e6, sy_um * 1e6} um")

            data_array = numpy.array(xdata)
            data_min = float(numpy.min(data_array))
            data_max = float(numpy.max(data_array))
            data_range = data_max - data_min

            data_uint8 = ((data_array - data_min/ data_range * 255).astype(numpy.uint8))

            img = Image.fromarray(data_uint8)
            #export_path = Path(r"C:\Users\Elizabeth.Wylie\Pictures\overview-scan.jpg")
            export_path = Path(r"C:\AS2\AS2User\Pictures\overview-scan.jpg")
            if not export_path.parent.exists():
                export_path.parent.mkdir(parents=True, exist_ok=True)

            img.save(export_path)

        except Exception as e:
            self._append_output(f"Failed to publish result: {e!r}")
            self.cancel_enabled.value = False
            return

        try:
            cartridge_result = stem_controller._get_rest_api("/exchange?property=CartridgeInStage")
            if cartridge_result.is_valid:
                cartridge_string = cartridge_result.value
                self._append_output_threadsafe(f"Cartridge in stage: {cartridge_string}")

                properties: JSONDict = {"ImageScaleRad_m": total_image_height, "ImageOffsetX_px": sx_um / pixel_size_m, "ImageOffsetY_px": sy_um / pixel_size_m, "ImageFile": str(export_path)}

                # Set the values on the cartridge

                stem_controller._put_rest_api(f"/exchange/cartridges/{cartridge_string}", content=properties)
                if hasattr(cartridge_result, "is_valid") and not cartridge_result.is_valid:
                    self._append_output_threadsafe(f"PUT failed: {cartridge_result.exception}")
            else:
                self._append_output_threadsafe(f"Failed to get CartridgeInStage: {cartridge_result.exception}")
                return

        except Exception as e:
            self._append_output(f"Failed to update cartridge data: {e!r}")
            self.cancel_enabled.value = False
            return

    def handle_perform_acquisition_clicked(self, widget: typing.Any) -> None:
        try:
            width_um = int(self.width_value)
            height_um = int(self.height_value)
            defocus_nm = int(self.defocus) * 1e-9
            reduce = int(self.binning)
        except ValueError:
            self._append_output("Please enter width, height, binning and defocus as integers.")
            return
        if width_um < 1 or height_um < 1 or reduce < 1:
            self._append_output("Please ensure width and height are positive.")
            return
        if width_um >= 1000 or height_um >= 1000:
            self._append_output("Warning: Requested scan size is outside of sensible limit")
            return
        if abs(defocus_nm * 1e9) < 1000 or abs(defocus_nm * 1e9) > 500000:
            self._append_output("Warning: Requested defocus is outside of sensible limit")
            return

        if self._acq_task and not self._acq_task.done():
            self._append_output("Acquisition already running.")
            return

        stem_controller = self.stem_controller
        camera = self.camera
        target_width_um = (width_um, height_um)

        self._acq_task = self._event_loop.create_task(
            self._run_acquisition_async(stem_controller, camera, defocus_nm, target_width_um, reduce)
        )
        self.cancel_enabled.value = False

    def handle_clear_minimap_clicked(self, widget: typing.Any) -> None:
        stem_controller = self.stem_controller
        try:
            cartridge_result = stem_controller._get_rest_api("/exchange?property=CartridgeInStage")
            if cartridge_result.is_valid:
                cartridge_string = cartridge_result.value
                properties: JSONDict = {"ImageScaleRad_m": 0.0, "ImageOffsetX_px": 0.0, "ImageOffsetY_px": 0.0, "ImageFile": ""}
                stem_controller._put_rest_api(f"/exchange/cartridges/{cartridge_string}", content=properties)
                self._append_output_threadsafe("Minimap cleared.")
            else:
                self._append_output_threadsafe(f"Failed to get CartridgeInStage: {cartridge_result.exception}")
        except Exception as e:
            self._append_output(f"Failed to clear minimap data: {e!r}")


class OverviewScanPanel(Panel.Panel):

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


class OverviewScanPanelExtension:

    extension_id = "overview-scan.panel"

    def __init__(self, api_broker: typing.Any) -> None:
        Registry.register_component(OverviewScanPanelUI(), {"overview-scan-panel"})
        Workspace.WorkspaceManager().register_panel(
            OverviewScanPanel,
            "overview-scan-main-panel",
            _("Overview Scan"),
            ["left", "right"],
            "right",
            {"panel_type": "overview-scan-panel"},
        )

    def close(self) -> None:
        pass
