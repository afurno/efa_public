import os.path
import re
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import qgis
from qgis.core import *
from processing.core.Processing import Processing
from processing.tools import *
from qgis.gui import QgsMapCanvas, QgsLayerTreeMapCanvasBridge, QgsMapCanvasLayer

NUM_CLASSES = 5
MAX_NUM_COMPONENTS = 1000
VARIABLE_ID_SHAPE = 'cell_id'
DELIMITER = ','

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def applyStyle(layer, classes, fieldName):
    symbol = QgsFillSymbolV2.createSimple({"outline_width": "0.15"})
    myStyle = QgsStyleV2().defaultStyle()
    ## setting ramp to Blues, first index of defaultColorRampNames
    colorRamp = myStyle.colorRamp("Spectral")
    label_format = QgsRendererRangeV2LabelFormat()
    label_format.setPrecision(1)
    label_format.setTrimTrailingZeroes(True)
    renderer = QgsGraduatedSymbolRendererV2.createRenderer(layer, fieldName, classes,
                                                           QgsGraduatedSymbolRendererV2.Jenks, symbol, colorRamp, legendFormat=label_format)
    renderer.setLabelFormat(label_format)
    renderer.updateColorRamp(colorRamp, inverted=True)
    layer.setLayerTransparency(50)
    layer.setRendererV2(renderer)

def visit_children(treeNode, layer_id):
    for child in treeNode.children():
        if isinstance(child, QgsLayerTreeLayer) and child.layerId() == layer_id:
            print "Found layer with id " + str(layer_id)
        else:
            print "Ignoring child with name " + str(child.name())
            QgsLegendRenderer.setNodeLegendStyle(child, QgsComposerLegendStyle.Hidden)
            visit_children(child, layer_id)
    
def printActiveLayer(title, layer, folder, filename):
    the_group = QgsProject.instance().layerTreeRoot()
    iface.mapCanvas().refresh()
    composerLegend.modelV2().setRootGroup(the_group)
    
    #composerLegend.model().setLayerSet([layer.id()])
    visit_children(composerLegend.modelV2().rootGroup(), layer.id())
    
    composerLegend.adjustBoxSize()
    maprender = iface.mapCanvas().mapRenderer()
    map_item.zoomToExtent(maprender.extent())
    composition.refreshItems()
    legendSize = composerLegend.paintAndDetermineSize(QPainter())
    composerLegend.setItemPosition(PAGE_WIDTH-legendSize.width(), 0)
    composerLegend.synchronizeWithModel()
    composerLegend.update()
    image = composition.printPageAsRaster(0)
    composition.exportAsPDF(folder + "/pdf/" + filename + '.pdf')
    image.save(folder + "/" + filename + '.png','png')
    
def joinResultsLayer(path, service_category, groupIndex, grid_layer_filename, results_layer, \
                     method_name, component_name, num_classes, variable_id_csv, elements_to_look_for, basepath_for_pics, save_pictures=False):
    grid_layer = QgsVectorLayer(grid_layer_filename, method_name + "_" + component_name + ' ' + elements_to_look_for, 'ogr')
    VORONOI_FILENAME_PATH = path + '/layers/' + service_category
    if not os.path.exists(VORONOI_FILENAME_PATH):
        os.makedirs(VORONOI_FILENAME_PATH)
    voronoi_filename = VORONOI_FILENAME_PATH + method_name + "_" + component_name + "_voronoi.shp"

    shpField = VARIABLE_ID_SHAPE
    csvField = variable_id_csv
    joinObject = QgsVectorJoinInfo()
    joinObject.joinLayerId = results_layer.id()
    joinObject.joinFieldName = csvField
    joinObject.targetFieldName = shpField
    joinObject.memoryCache = True
    joinObject.prefix = ""
    joinObject.setJoinFieldNamesSubset([component_name])
    grid_layer.addJoin(joinObject)
    
    if grid_layer.isValid():
        crs = grid_layer.crs().authid() # without () after originalLayer
        inLayerGeometryType = ['Point','Line','Polygon'][grid_layer.geometryType()]
        tempLayer = QgsVectorLayer(inLayerGeometryType + "?crs={0}".format(crs), "temporary_points", "memory")
        
        expr = QgsExpression( "\"" + component_name + "\"!='None'" )
        features = list(grid_layer.getFeatures(QgsFeatureRequest( expr )))
        
        temp_data = tempLayer.dataProvider()
        attr = grid_layer.dataProvider().fields().toList()
        temp_data.addAttributes(attr)
        tempLayer.updateFields()
        temp_data.addFeatures(features)
        tempLayer.addJoin(joinObject)
        QgsMapLayerRegistry.instance().addMapLayer(tempLayer)
        
        general.runalg("qgis:voronoipolygons", tempLayer, 1, voronoi_filename)
        voronoi_layer = QgsVectorLayer(voronoi_filename, method_name + "_" + component_name + ' ' + elements_to_look_for, 'ogr')
        if voronoi_layer.isValid():
            applyStyle(voronoi_layer, num_classes, component_name)
        else:
            print "Problem in Voronoi layer creation (" + method_name + "_" + component_name + ")"

        voronoi_layer.setLayerName(component_name)
        QgsMapLayerRegistry.instance().addMapLayer(voronoi_layer)
        QgsMapLayerRegistry.instance().removeMapLayer(tempLayer)
        
        if not os.path.exists(basepath_for_pics + "/" + elements_to_look_for + "_pictures/"):
            os.makedirs(basepath_for_pics + "/" + elements_to_look_for + "_pictures/")
        
        if not os.path.exists(basepath_for_pics + "/" + elements_to_look_for + "_pictures/pdf/"):
            os.makedirs(basepath_for_pics + "/" + elements_to_look_for + "_pictures/pdf/")
        if save_pictures:
            printActiveLayer(component_name, voronoi_layer, basepath_for_pics + "/" + elements_to_look_for + "_pictures/", method_name + "_" + component_name + ' ' + elements_to_look_for)
            QgsMapLayerRegistry.instance().removeMapLayer(voronoi_layer)
        
        if groupIndex:
            toc = iface.legendInterface()
            toc.moveLayer(voronoi_layer, groupIndex)
            legend = iface.legendInterface()  # access the legend
            legend.setLayerVisible(voronoi_layer, False)
        #legend.setLayerVisible(voronoi_layer, False)
    else:
        print "PROBLEM: cannot apply style and create layer"


def createResultsLayer(results_filename, groupIndex, results_layer_name, delimiter):
    results_uri = 'file:///%s?delimiter=%s' % (results_filename, delimiter)
    results_layer = QgsVectorLayer(results_uri, results_layer_name, 'delimitedtext')
    QgsMapLayerRegistry.instance().addMapLayer(results_layer)
    if groupIndex:
        toc = iface.legendInterface()
        toc.moveLayer(results_layer, groupIndex)
        legend = iface.legendInterface()  # access the legend
        legend.setLayerVisible(results_layer, False)
    results_file = open(results_filename, "r")
    splitted_line = results_file.readline().strip().split(",")
    return results_layer, splitted_line[1:]

def plot_csv_file(a_file, element_to_look_for, basepath, root_group, save_pictures=False):
    if "signatures" in basepath.split("/")[-1] and PLOT_SIGNATURES:
        variable_id_csv = "index"
        element_to_look_for = "norm_signatures"
    else:
        variable_id_csv = VARIABLE_ID_CSV
    if a_file.endswith(".csv") and a_file.startswith(element_to_look_for):
        method_name = basepath.split("/")[-1]
        groupIndex = None
        group_layer_name = method_name + "_" + element_to_look_for
        if root_group is not None:
            root_group.insertGroup(0, group_layer_name)
            toc = iface.legendInterface()
            groups = toc.groups()
            groupIndex = groups.index(group_layer_name)
        data_filepath = os.path.join(basepath, a_file)
        print "Analyzing file " + data_filepath
        results_layer, component_names = createResultsLayer(data_filepath, groupIndex, group_layer_name, DELIMITER)

        for component_name in sorted(component_names[:MAX_NUM_COMPONENTS], key=natural_keys, reverse=True):
            joinResultsLayer(path, "", groupIndex, GRID_LAYER_FILENAME, results_layer, group_layer_name, component_name, NUM_CLASSES, variable_id_csv, element_to_look_for, basepath, save_pictures=save_pictures)
        if save_pictures:
            QgsMapLayerRegistry.instance().removeMapLayer(results_layer)
    else:
        data_filepath = os.path.join(basepath, a_file)
        print "Skipping file " + data_filepath


composers = iface.activeComposers()
composition = composers[0].composition()
map_item = composition.getComposerItemById('map0')
PAGE_WIDTH = map_item.rect().width()
composerLegend = composition.getComposerItemById('legend0')
if composerLegend is None:
    print "Creating new legend"
    composerLegend = QgsComposerLegend(composition)
    composerLegend.setId("legend0")
else:
    print "Found existing legend"
composition.addComposerLegend(composerLegend)
composerLegend.setTitle("")
composerLegend.setAutoUpdateModel(False)
composerLegend.setEqualColumnWidth(True)
composerLegend.setBackgroundEnabled(True)
#composerLegend.setBackgroundColor(QColor('white'))
composerLegend.setFrameEnabled(False)
#composerLegend.setFrameOutlineColor(QColor('red'))
#composerLegend.setFrameOutlineWidth(1)
#composerLegend.setTitle(title)
composerLegend.setStyleFont(QgsComposerLegendStyle.Title, QFont("Times", 0))
composerLegend.setStyleFont(QgsComposerLegendStyle.Group, QFont("Times", 12))
#composerLegend.setStyleFont(QgsComposerLegendStyle.Subgroup, None)

composerLegend.setStyleFont(QgsComposerLegendStyle.SymbolLabel, QFont("Times", 12))
composerLegend.setColumnSpace(1)
composerLegend.setSymbolHeight(2)
composerLegend.setSymbolWidth(3)

Processing.initialize()
prjfi = QFileInfo(QgsProject.instance().fileName())
path = prjfi.absolutePath()  # what you are probably looking for
qfd = QFileDialog()

title = 'Select GRID LAYER...'
GRID_LAYER_FILENAME = QFileDialog.getOpenFileName(qfd, title, path)
DATA_BASEPATH = QFileDialog.getExistingDirectory(qfd, "Open Directory", ".", options = QFileDialog.ShowDirsOnly)

if "space2" in DATA_BASEPATH:
    VARIABLE_ID_CSV = 'cell_id'
    PLOTTED_ELEMENTS = "scores" 
    PLOT_SIGNATURES = True
elif "time2" in DATA_BASEPATH:
    VARIABLE_ID_CSV = 'cell_id'
    PLOTTED_ELEMENTS = "loadings"
    PLOT_SIGNATURES = False

FOLDERS_TO_EXPLORE = ["minres_varimax", "ml_varimax"]

for folder in FOLDERS_TO_EXPLORE:
    root = QgsProject.instance().layerTreeRoot()
    #root_group = root.insertGroup(0, folder)
    for basepath, dirs, files in os.walk(DATA_BASEPATH + "/" + folder):
        for a_file in sorted(files, reverse=True):
            plot_csv_file(a_file, PLOTTED_ELEMENTS, basepath, None, save_pictures=True)

for folder in FOLDERS_TO_EXPLORE:
    root = QgsProject.instance().layerTreeRoot()
    root_group = root.insertGroup(0, folder)
    for basepath, dirs, files in os.walk(DATA_BASEPATH + "/" + folder):
        for a_file in sorted(files, reverse=True):
            plot_csv_file(a_file, PLOTTED_ELEMENTS, basepath, root_group)