import numpy as np
import imageio
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import easygui as eg

def pic_height(figname):
    """
    Open figure, check fig size using numpy, and return the height in pixels.
    figname = Provide figure name with extension
    """
    #Load an image in grayscale
    img = imageio.imread(figname)
    #calculate pic height and total area and display for user
    pic_height = img.shape[0]
    pic_total_area = img.shape[0]*img.shape[1]
    print("Picture height =", pic_height)
    return pic_height
    
def fig_total_area(figname):
    """
    Open the provided image using imageio, calculates the shape with numpy, and multiply height x lenght to obtain picture size in pixels.
    figname = name of the figure containing the extension (tif, png).
    """
    #Load an image in grayscale
    img = imageio.imread(figname)
    pic_total_area = img.shape[0]*img.shape[1]
    print("Total area in pixels =", pic_total_area)
    return pic_total_area
    
def fig_micrometers():
    """
    Function that asks for objective size used to acquire images with Cytation 5. The values here provided are specifically for Cytation 5 objectives - Biotek. Takes user input only.
    """
    answer = eg.enterbox(msg="'Which objective size have you used? Options are 4x, 10x and 20x.'", default='', strip=True)
    objective_conversion = 0
    if answer == '4x':
        objective_conversion = 1.611928
        print('Objective conversion from um to pixels is',objective_conversion,'micrometers.')
    elif answer == '10x':
        objective_conversion = 0.644608
        print('Objective conversion from um to pixels is',objective_conversion,'micrometers.')
    elif answer == '20x':
        objective_conversion = 0.321895
        print('Objective conversion from um to pixels is',objective_conversion,'micrometers.')
    else:
        print("Write again; the options are 4x, 10x or 20x.")
    
    return objective_conversion

def open_layout():
    """
    Have a folder with the date of the assay (YYYYMMDD), and a csv file with metadata of the assay based on the date of the assay (YYYYMMDD) + "metadata.csv". Or provide path within the name of the file: for example, "20181102/20181102_metadata.csv"  
    """
    path_files = eg.fileopenbox(msg="CHOOSE LAYOUT FILE", default='*', filetypes=None, multiple=False)
    date = eg.enterbox(msg="PROVIDE DATE WHEN THE ASSAY WAS PERFORMED", default='', strip=True)
    df = pd.read_csv(path_files, sep = ',')
    df.head(20)
    return df, date

def add_columns(df):
    """
    Add columns to the layout dataframe
    """
    df['Slope'] = float(0) #slope for each curve
    df['Intercept'] = float(0) #intercept in the y axis
    df['R^2 score'] = float(0) #r2 of the fitting
    df['Velocity'] = float(0) #velocity calculated from 
    df['Time'] = float(0) #create time column
    df['Begin'] = float(0) #beggining and end 
    df['End'] = float(0.60) #takes 60% of the data to fit linear regression
    return df

def migration_velocity(layout, index, pic_height, objective_conversion):
    """"
    Function that calculates velocity of cell migration using
    slope: coefficient obtained from linear regression
    pic_height: the height of your picture in pixels
    objective_conversion: objective factor from the objective used in the image acquisition. Confirm this value with your manufacturer
    """
    velocity = ((layout.at[index,'Slope']/pic_height)/2)*objective_conversion
    layout.at[index,'Velocity'] = velocity
    return velocity

def time_velocity(layout, index, pic_total_area):
    """time it takes for scratch to be closed. based on the regression lines previously calculated
    """
    xtime = ((pic_total_area - layout.at[index,'Intercept'])/layout.at[index,'Slope'])/60
    layout.at[index,'Time'] = xtime
    return xtime

def open_df(well, pic_total_area, suffix = "_data", area_column = "Area (pixel^2)", path_files = 'Example'):
    """
    Open dataframe, create cell-covered column, and return df.
    suffix = "_data" is standard, change if different.
    area_column = "Area (pixel^2) standard. Provide the column that has the scratch measured area in pixels. 
    """
    df = pd.read_csv(path_files + '/' + well + suffix + ".csv", sep="\t|:|;", encoding = "unicode_escape", engine = "python")
    cell_covered = pic_total_area - df[area_column]
    df["cell covered"] = cell_covered
    return df

def define_end(df, layout, index):
    """
    Defines where to fit linear regression curve (how much of the data to fit).
    The standard is 0.6 (60%, "End" column). This value can be changed using the function change_fit(). 
    """
    #interpolation to find the range between 0-1 inside cell-covered column
    interpolate_df = interp1d([0, 1],[df['cell covered'].min(), df['cell covered'].max()]) 
    #find the value in cell-covered that corresponds to 60% of the data (assuming the higher value is equal to 100%)
    list_linearfit = []
    for value in df['cell covered']:
        if value <= interpolate_df(layout['End'][index]):
            list_linearfit.append(value)
    begin = 0
    end = len(list_linearfit) #ends in the value that represents 60% of the data or another value defined by the user
    return begin, end
          
def fit_linear_regression(df, layout, begin, end, index, time_column = "Time (min)"):
    """
    Fit linear regression. create X and y, transform the columns to numpy and reshape. Add the calculated parameters (slope, score, and intercept) into the layout dataframe. 
    time_column = provide which is the name of the column containing the information about time between pictures. 
    """
    y = df['cell covered'][begin:end].to_numpy().reshape(-1, 1)
    # print(y, 'y')
    X = df[time_column][begin:end].to_numpy().reshape(-1, 1)
    # print(X, 'x')
    reg = LinearRegression().fit(X, y) #create and fit linear regression function
    #df.at: Access a single value for a row/column label pair, and save the variables into the df layout
    layout.at[index,'Slope'] = reg.coef_ #slope
    layout.at[index,'R^2 score'] = reg.score(X, y) #score
    layout.at[index,'Intercept'] = reg.intercept_ #intercept
    return X, y, reg

def calculate_fit(df, layout, begin, end, index, time_column = "Time (min)"):
    """
    From linear regression, create linear curve to fit data and plot above the data points.
    """
    x_fit = df[time_column][begin:end].to_numpy()
    y_fit = ((layout.at[index,'Slope'])*x_fit) + layout.at[index,'Intercept']
    return x_fit, y_fit
    
def output_folder(path_files = 'Example'):
    directory = "output"
    path = os.path.join(path_files, directory)
    os.makedirs(path, exist_ok=True)
    return path

def plot(df, layout, begin, end, index, date, well, time, pic_total_area, x_fit, y_fit, time_column = "Time (min)", output_path = 'Example'):
    """
    Define variables and where to plot each information, then generate a plot for each well.
    """
    x_text = 0.5
    y_text = 0.5
    interpolate_x = interp1d([0, 1],[df[time_column].min(), df[time_column].max()])
    interpolate_y = interp1d([0, 1],[df['cell covered'].min(), df['cell covered'].max()])
    textstr = 'Slope' + '=%.2f' % (layout.at[index,'Slope'])
    plt.cla()
    plt.clf()
    plt.plot(df[time_column],df['cell covered'],'y.', label='Sample')
    plt.plot(x_fit,y_fit, 'r-', linewidth=3, label='Fitting')
    plt.plot((time*60), pic_total_area, 'g*', label = 'Estimated time')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(interpolate_x(x_text), interpolate_y(y_text), textstr, fontsize=14, verticalalignment='top', bbox=props) #find the place to put the box with slope value where x_text is in 80% percent of data and y_text 50% of data
    plt.legend(loc='best')
    plt.xlabel('Time(min)')
    plt.ylabel('Cell-covered area')
    plt.title(layout['Cell'][index] + " " + layout['Treatment'][index])
    plt.savefig(output_path + '/layout_' + well + '.png', dpi = 150, bbox_inches='tight')
    plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')
    return 
        
def change_end_fit(layout):

    """
    Change values in the layout column "End" based on the index and value provided here.
    """
    indexes = input("Enter indexes from the wells you wish to change separated by space")
    change_to = input("Enter the value you'd like to change the end of the fitting (from 0.6 to?)")
    user_list = indexes.split()
    #convert each item to int type
    for i in range(len(user_list)):
        user_list[i] = int(user_list[i]) #convert each item to int type
    for end in user_list:
        layout.at[end,"End"] = change_to
        
def plot_all(layout, hueorder, date, xaxis = "Cell", yaxis = "Velocity", ylimit = (0,1), ylabel = "Cell migration velocity \u03bcm/min", output_path = 'Example'):
    """
    Plot all the datapoints belonging to that assay. 
    xaxis and yaxis = provide which columns to plot
    ylimit = change the y limit in the axis
    \u03bc = stands for micro greek symbol
    """
    #first, organize the names of the treatments in alphabetic order
    layout.columns = layout.columns.str.replace(' ', '')
    #hueorder = sorted(list(layout['Treatment'].unique()))  
    sns.set(font_scale=2)
    g = sns.catplot(x = xaxis, y = yaxis, kind = "violin", hue = ("Treatment"), height = 10, aspect = 1.5, hue_order = hueorder, data = layout, palette = 'husl') 
    plt.ylim(ylimit)
    plt.xlabel("Cells", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    g.fig.suptitle(ylabel,y=1.1,fontsize=25)
    g._legend.set_title("Treatments")
    plt.show()
    g.savefig(output_path + "/" + yaxis + date + '.tiff')
    g.savefig(output_path + "/" + yaxis + date + '.svg')
    sns.reset_orig()
     

def import_results(output_path = 'Example'):

    """
    Import results calculated during processing scratch assay to plot in another notebook.
    """
    date = input('Which is the date of your experiment (YYYYMMD)? Provide the path if different from YYYYMMDD')
    df = pd.read_csv(output_path + '/Results_' + date + '.csv', sep = ',')
    df.head(20)
    return df, date
    
def plot_all_replicates(layout, hueorder, xaxis = "Cell", yaxis = "Velocity", ylimit = (0,1), ylabel = "Cell migration velocity \u03bcm/min", assay = "Folder_to_export"):
    """
    Plot all the datapoints belonging to that assay. 
    xaxis and yaxis = provide which columns to plot
    ylimit = change the y limit in the axis
    \u03bc = stands for micro greek symbol
    """
    #first, organize the names of the treatments in alphabetic order
    layout.columns = layout.columns.str.replace(' ', '')
    #hueorder = sorted(list(layout['Treatment'].unique()))  
    sns.set(font_scale=2)
    g = sns.catplot(x = xaxis, y = yaxis, kind = "box", hue = ("Treatment"), height = 10, aspect = 1.5, hue_order = hueorder, data = layout, palette = 'husl') 
    plt.ylim(ylimit)
    plt.xlabel("Cells", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    g.fig.suptitle(ylabel,y=1.1,fontsize=25)
    g._legend.set_title("Treatments")
    plt.show()
    g.savefig(assay + "/" + yaxis + '.tiff')
    g.savefig(assay + "/" + yaxis + '.svg')
    sns.reset_orig()