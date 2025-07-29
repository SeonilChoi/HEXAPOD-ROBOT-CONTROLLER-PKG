from robot_rqt.robot_rqt_widget import RobotRQtWidget
from rqt_gui_py.plugin import Plugin

class RobotRQtController(Plugin):
    def __init__(self, context):
        super().__init__(context)

        self.setObjectName('RobotRQtController')
        self.widget = RobotRQtWidget(context.node)
        serial_number = context.serial_number()
        if serial_number >= 1:
            self.widget.setWindowTitle(self.widget.windowTitle() + ' ({0})'.format(serial_number))
        context.add_widget(self.widget)

    def shutdown_plugin(self):
        print("\n[RobotRQtController::shutdown_plugin] Shutdown the Robot RQt Controller.")
        self.widget.shutdown_widget()
