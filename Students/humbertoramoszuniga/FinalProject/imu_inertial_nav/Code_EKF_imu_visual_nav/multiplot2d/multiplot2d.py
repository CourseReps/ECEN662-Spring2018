import os
import numpy as np
import matplotlib.pyplot as plt


class MultiPlotter:

    def __init__(self, subplot_grid, size_inches=None, name="figure",
                 subplot_args=dict()):
        """
        matplotlib wrapper for easily creating subplots.

        This class provides a consistent interface for adding data and plot
        elements to a series of subplots. It also offers easy access to the
        underlying figure and subplot(s).

        For best results, run plt.ioff() before creating a multiplotter.

        Parameters
        ----------
        subplot_grid : {list, tuple, int}
            If `subplot_grid` is int n, then multiplotter will create an nx1
            grid of subplots.
            If `subplot_grid` is a list or tuple of integers of length 2 [n,m],
            multiplotter will create an nxm grid of subplots.
            If `suplot_grid` is a list or tuple of AxesSubplot objects,
            multiplotter will use these subplots instead of creating new ones.
        size_inches : tuple of integers, optional
            Width, height in inches of the figure. This is particuarlly
            important for saving figures. If the dpi of the monitor is the not
            the same as the dpi of the figure, the figure size may not match
            this physical size onscreen. If no argument is specified, the
            figure will be sized such that each subplot is 3"x3".
        name : str, optional, default: "figure"
            `name` will be the name of the figure window. It will also be the
            default name for other things like filenames.
        subplot_args : keyword arguments, optional, def
            keyword arguments for matplotlib.pyplot.subplot
            (http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)

        Notes
        -----
        For complex subplot arrangements, create the subplots using
        subplot2grid, GridSpec, or another method
        (http://matplotlib.org/users/gridspec.html), put them into a list, an
         pass them as `subplot_grid`.

        For simple usage (creating an nxm grid), just pass a list or tuple
        [n,m] as `subplot_grid`.

        To see a list of all multiplotter methods, type help(multiplotter) in
        the interpreter.
        """

        # case 1: subplot_grid is the number of cols of subplots
        if type(subplot_grid) == int:
            subplot_grid = [1, subplot_grid]
            self._grid_shape = subplot_grid

        # case 2: subplot_grid is a list of the number of rows and columns,
        # respectively, of subplots
        if ((type(subplot_grid) == list or type(subplot_grid) == tuple) and
           type(subplot_grid[0]) == int and len(subplot_grid) == 2):

            if subplot_grid[0] == 0:
                subplot_grid[0] = 1
            if size_inches is None:
                size_inches = [subplot_grid[1]*3, subplot_grid[0]*3]
            self._figure, plot_list_grid = plt.subplots(*subplot_grid,
                                                        figsize=size_inches,
                                                        **subplot_args)
            self._grid_shape = subplot_grid

        # case 3: subplot_grid is a sequence of supblots items
        else:
            # assume subplot_grid is a sequence of supblots items
            try:
                self._figure = subplot_grid[0].figure
                plot_list_grid = np.array(subplot_grid)
                # print("TODO: figure out grid shape")
            # if that didn't work, print error message and return
            except:
                print("\nsubplot_grid must be a list of grid dimensions")
                print("or a list of subplots.")
                print("No subplots have been created.\n")
                self._plot_list = []
                self._all_plot_indexes = []
                self._figure = plt.figure(figsize=size_inches)
                return

        # set window title
        self._figure.canvas.set_window_title(name)

        # ensure that plot_list_grid is iterable
        # this is necessary for the case of only one subplot
        if type(plot_list_grid) != np.ndarray:
            plot_list_grid = np.array([plot_list_grid])

        # at this point, plot_list_grid should be a numpy array
        # it either contains a sequence of subplot items, or a sequence of
        # numpy arrays that themselves contain a sequence of suplot items

        # save each plot in row major order
        self._plot_list = []
        for row in plot_list_grid:
            if type(row) == np.ndarray:
                for col in row:
                    self._plot_list.append(col)
            else:
                self._plot_list.append(row)

        self._all_plot_indexes = range(len(self._plot_list))
        # see _prepare_fig_for_display() method for where these are used
        self._tighten_layout = True
        # by default, don't shrink from top
        self.shrink_top_inches = 0.0

    def add_data(self, target_plots, x, y, labels=None, line_styles=None,
                 plots_share_data=False):
        """
        Plot data on one or more subplots.

        Data can be plotted using optional arguments for labels and line
        styles.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        x : ndarray
            x-coordinates of data in `y`.
        y : ndarray
            Data to plot.
        labels : string or tuple of strings, optional
            The items in `labels` will be used to populate the subplot legend.
            The default argument is not explicit in the function definition;
            however, if no argument is given, the ith dataset will receive a
            label of "data i".
        line_styles : tuple of keyword arguments, optional
            Keyword arguments for matplotlib.lines
            (http://matplotlib.org/api/lines_api.html)
            Default is empty dictionary, which means that default styles will
            be used.
        plots_share_data : boolean, optional, default: False
            See Notes.

        Returns
        -------
        line_list : list of list of matplotlib.lines.Line2D items
            A list of the lines that were added to each subplot. The ith item
            in `line_list` corresponds to the ith subplot in `target_plots`.
            The kth item of the ith item corresponds to the kth plotted line.
            So, if 1 dataset was added to 5 plots, len(`line_list`)==5 and
            len(`line_list`[0])==1.If 3 datasets were added to 5 plots,
            len(`line_list`)==5 and len(`line_list`[0])==3.

        Notes
        -----
        `x` can be a vector, even if `y` is not. In this case, all datasets
        in `y` will be plotted against vector `x`. `x` and `y` must have
        the same length.

        If `y` is a vector, `x` must be a vector. `x` and `y` must have
        the same length.

        If `x` is not a vector, it must have the same number of columns as
        `y`. `x` and `y` must have the same number of rows.

        If the number of items in `target_plots` does not equal the number of
        columns in `y`, all datasets will be plotted on every plot in the
        `target_plots` sequence.

        For the case where the number of items in `target_plots` equals
        the number of columns in `y`, this function assumes that the datasets
        should be distributed, and not shared, among the subplots indicated in
        `target_plots`. If it is desirable that each subplot show every
        dataset, `plots_share_data` must be True.

        The `labels` and `line_styles` work in a similiar way. If one item is
        entered, it will be used for all datasets. If more than one item is
        inputted, the number of items must equal or exceed the number of
        datasets. If the number of items exceeds the number of datasets,
        only the first n items will be used, where n is the number of
        datasets.

        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        shape_x = np.shape(x)
        # make 1D array x into a column vector
        if len(shape_x) == 1:
            x = np.array([x]).T
            shape_x = np.shape(x)
        # make row vector x into a column vector
        elif shape_x[0] == 1:
            x = x.T
            shape_x = np.shape(x)

        shape_y = np.shape(y)
        # make 1D array y into a column vector
        if len(shape_y) == 1:
            y = np.array([y]).T
            shape_y = np.shape(y)
        # if y is a vector, it must be a column vector
        elif shape_y[0] == 1:
            y = y.T
            shape_y = np.shape(y)

        xrows = shape_x[0]
        xcols = shape_x[1]
        yrows = shape_y[0]
        ycols = shape_y[1]
        # x and y must have the same number of rows
        if xrows != yrows:
            print("\nx and y must have the same number of rows.\n")
            return -1
        # if x is a matrix, it's shape must match y
        if xcols != 1 and xcols != ycols:
            print("\nx must be a vector or of the same shape as y.\n")
            return -1

        if labels is None:
            labels = ["data " + str(i+1) for i in range(ycols)]
            # labels = ["data " + str(i+1) for i in range(ycols)]
        elif type(labels) == str:
            labels = [labels]*ycols
            # labels = [labels + " " + str(i+1) for i in range(ycols)]
        elif len(labels) == 1:
            labels = [labels[0]]*ycols
            # labels = [labels[0] + " " + str(i+1) for i in range(ycols)]
        elif len(labels) < ycols:
            print("\nThe number of cols in data matrix y is %d." % ycols)
            print("%d labels were received. There must be one" % len(labels))
            print("label for every col in data matrix y.\n")
            return -1

        if line_styles is None:
            line_styles = [dict()]*ycols
        # only one line style inputted
        # this one style will be used for every dataset
        elif type(line_styles) == dict:
            line_styles = [line_styles]*ycols
        elif len(line_styles) == 1:
            line_styles = line_styles*ycols
        elif len(line_styles) < ycols:
            print("\nThe number of cols in data matrix y is %d." % ycols)
            print("%d line styles were received. There must be one" % len(
                line_styles))
            print("line style for every col in data matrix y")
            print("or one line style which will be shared by all lines.\n")
            return -1

        if num_target_plots != ycols:
            plots_share_data = True

        # the ith item in line_list corresponds to the ith subplot
        # the kth item the ith item corresponds to the kth plotted line
        line_list = [[] for i in range(len(target_plots))]

        # if all the conditions are satisfied, plot the data
        for plot_counter, plot_id in enumerate(target_plots):
            # all the datasets will be put on this plot
            if plots_share_data is True:
                data_for_this_plot = y
                labels_for_this_plot = labels
                line_styles_for_this_plot = line_styles

            # only one dataset will be put on this plot
            else:
                data_for_this_plot = y[:, plot_counter:plot_counter+1]
                labels_for_this_plot = [labels[plot_counter]]
                line_styles_for_this_plot = [line_styles[plot_counter]]

            data_set_index = 0
            iterate_over = zip(data_for_this_plot.T, labels_for_this_plot,
                               line_styles_for_this_plot)
            for plot_ydata, label, line_style in iterate_over:
                if xcols == 1:
                    plot_xdata = x.T
                else:
                    plot_xdata = x.T[data_set_index]
                plot_item = self._plot_list[plot_id]
                line_list[plot_counter].append(plot_item.plot(
                    plot_xdata.flatten(), plot_ydata.flatten(), label=label,
                    **line_style)[0])
                data_set_index += 1

        return line_list

    def set_plot_titles(self, target_plots, plot_titles,
                        plot_title_args=dict()):
        """
        Set the title of one or more subplots.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        plot_titles : string or tuple of strings
            Title(s) to set. If only one title is provided, all target
            plots will receive the same title. Otherwise, there should be one
            title for each target plot.
        plot_title_args : keyword arguments, optional
            Keyword arguments for matplotlib.axes.Axes.set_title
            (http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_title).

        Returns
        -------
        title_list : list of matplotlib.text.Text items
            A list of the subplot titles that were added to each subplot. The
            ith item in `title_list` corresponds to the title of the ith
            subplot in `target_plots`.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        # turn plot_titles into lists if they are not
        if type(plot_titles) == str:
            plot_titles = [plot_titles]

        len_plot_titles = len(plot_titles)
        title_list = []
        if len_plot_titles != 1 and len_plot_titles < num_target_plots:
            print("\n%d plot titles were received." % len_plot_titles)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one title for every plot")
            print("or only one title which will be shared by all plots.\n")
            return -1

        for plot_counter, plot_id in enumerate(target_plots):
            if len_plot_titles == 1:
                title = plot_titles[0]
            else:
                title = plot_titles[plot_counter]

            title_list.append(
                self._plot_list[plot_id].set_title(title, **plot_title_args))

        return title_list

    def set_axis_titles(self, target_plots, x_titles="", y_titles="",
                        axis_title_args=dict()):
        """
        Set the axis titles of one or more subplots.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        x_titles : string or tuple of strings, optional, default: ""
            X-axis label(s) to set. If only one label is provided, all
            target plots will receive the same x-label. Otherwise, there should
            be one label for each target plot. If the label for any target plot
            is set to an empty string (i.e. '') then the label will not be
            changed for that plot.
        y_titles : string or tuple of strings, optional, default: ""
            Y-axis label(s) to set. If only one label is provided, all
            target plots will receive the same y-label. Otherwise, there should
            be one label for each target plot. If the label for any target plot
            is set to an empty string (i.e. '') then the label will not be
            changed for that plot.
        axis_title_args : keyword arguments, optional
            Keyword arguments for matplotlib.axes.Axes.set_xlabel
            (http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_xlabel)
            and matplotlib.axes.Axes.set_ylabel
            (http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.set_ylabel).

        Returns
        -------
        axis_title_list : list of lists of matplotlib.text.Text items
            A list of the subplot axis titles that were added to each subplot.
            The ith item in `title_list` corresponds to the axis titles of the
            ith subplot in `target_plots`. The first item in the ith item in
            `title_list` corresponds to the x-axis title. The second item in
            the ith item in `title_list` corresponds to the y-axis title.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        # turn xlables and y_titles into lists if they are not
        if type(x_titles) == str:
            x_titles = [x_titles]
        if type(y_titles) == str:
            y_titles = [y_titles]

        len_x_titles = len(x_titles)
        len_y_titles = len(y_titles)
        if len_x_titles != 1 and len_x_titles < num_target_plots:
            print("\n%d labels for x-axis were received." % len_x_titles)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one label for every plot")
            print("or only one label which will be shared by all plots.\n")
            return -1

        if len_y_titles != 1 and len_y_titles < num_target_plots:
            print("\n%d labels for y-axis were received." % len_y_titles)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one label for every plot")
            print("or only one label which will be shared by all plots.\n")
            return -1

        axis_title_list = [["", ""]]*num_target_plots
        return_plot_index = 0

        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]

            if len_x_titles == 1:
                x_title = x_titles[0]
            else:
                x_title = x_titles[plot_counter]

            if len_y_titles == 1:
                y_title = y_titles[0]
            else:
                y_title = y_titles[plot_counter]

            if x_title != '':
                axis_title_list[return_plot_index][0] = (
                    plot_item.set_xlabel(x_title, **axis_title_args))
            if y_title != '':
                axis_title_list[return_plot_index][1] = (
                    plot_item.set_ylabel(y_title, **axis_title_args))
            return_plot_index += 1

        return axis_title_list

    def add_legend(self, target_plots, legend_args=dict()):
        """
        Add a legend to one or more plots.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        legend_args : keyword arguments, optional
            Keyword arguments for matplotlib.axes.Axes.legend
            (http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.legend)

        Returns
        -------
        legend_items : list of matplotlib.legend.Legend items
            A list of the legend items that were added to the subplots. The ith
            item in `legend_items` corresponds to the legend of the ith subplot
            in `target_plots`.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        num_target_plots = len(target_plots)
        if ret is False:
            return -1

        legend_items = []
        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]
            legend_items.append(plot_item.legend(**legend_args))

        return legend_items

    def add_grid(self, target_plots, grid_args=dict()):
        """
        Add a grid to one or more plots.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        grid_args : keyword arguments, optional
            Keyword arguments for matplotlib.axes.Axes.grid
            (http://matplotlib.org/api/axis_api.html)

        Returns
        -------
        grid_items : list of matplotlib.axes.Axes.grid items
            A list of the grid items that were added to the subplots. The ith
            item in `grid_items` corresponds to the grid of the ith subplot in
            `target_plots`.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        num_target_plots = len(target_plots)
        if ret is False:
            return -1

        grid_items = []
        for plot_counter, plot_id in enumerate(target_plots):
            grid_items.append(self._plot_list[plot_id].grid(**grid_args))

        return grid_items

    def get_plots(self, target_plots):
        """
        Get the underlying subplot ojbect instances for this multiplotter.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be return.

        Returns
        -------
        plot_list : list of matplotlib.axes._subplots.AxesSubplot items
            A list of the subplot items. The ith
            subplot in `plot_list` corresponds to the ith subplot in
            `target_plots`.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1

        plot_list = []
        for plot_id in target_plots:
            plot_list.append(self._plot_list[plot_id])

        return plot_list

    def scale_y_limits(self, target_plots, scale_high=1.05, scale_low=1.05):
        """
        Scale the limits of the y-axis by some amount.

        Matplotlib automatically scales the y-axis to fit all of the data on
        a plot. However, occasionally some of the data is obscured because
        it is on or very close to the top or bottom edge of the plot. This
        method will add space to the top or bottom of the plot to keep this
        from happening. The amount of space is relative to the current limits.

        Parameters
        ----------
        target_plots : sequence of integers
            Indexes, in row-major order, of the plots to be affected by this
            function.
        scale_high : float, optional, default: 1.05
            The new upper y-limit will be the current upper y-limit times
            `scale_high`.
        scale_low : float, optional, default: 1.05
            The new lower y-limit will be the current lower y-limit times
            `scale_low`.

        Returns
        -------
        y_limit_list : list of tuples
            A list of the new y-limits. The ith item in `y_limit_list`
            corresponds to the y-limits of the ith subplot in
            `target_plots`. The first item in the ith item in
            `y_limit_list` corresponds to the lower limit. The second item in
            the ith item in `y_limit_list` corresponds to the upper limit.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1

        y_limit_list = []
        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]
            # update axis data limits
            plot_item.relim()
            # current y limits (lower,upper)
            y_lim = plot_item.get_ylim()
            # the "middle bound"
            # the value of y "in between" the upper and lower limits
            y_mean = (y_lim[1]+y_lim[0])/2
            # the difference between the upper bound and the "middle bound"
            y_amp = (y_lim[1]-y_lim[0])/2

            # new low limit
            y_low = y_mean - scale_low*y_amp
            # new upper limit
            y_high = y_mean + scale_high*y_amp
            y_limit_list.append(plot_item.set_ylim(y_low, y_high))

        return y_limit_list

    def get_figure(self):
        """
        Return the underlying matplotlib figure of this multiplotter instance.

        Returns
        -------
        figure : matplotlib.figure.Figure
            The the underlying matplotlib figure of this multiplotter instance.
        """
        self._prepare_fig_for_display()
        return self._figure

    def save(self, filename=None, save_directory=os.getcwd(), save_args=dict(),
             fig_alpha=1.0, plot_alpha=1.0, pdf=None):
        """
        Save the figure.

        Before saving the figure, the plot and plot element spacing will be
        optimized to avoid plots or elements from intersecting each other.
        The figure will be saved at the size (in inches) specified when the
        multiplotter was initialized.

        Although there are many arguments, none of them are required.

        Parameters
        ----------
        filename : string, optional
            Name of the file to save. If not specified, the file will be given
            the name of the multiplotter instance (this was set on
            initialization). Don't include the file extension;
            instead pass the file extension you want as the value of the
            keyword "format" in the `save_args` dictionary (by default, the
            file will be a .png). In addition, do not include the directory;
            it's better to specify this using `save_directory`.
        save_directory : string, optional
            A relative or absolute file path where you want to save the image.
            If not specified, the file will be saved in the current working
            directory.
        save_args : keyword arguments, optional
            Keyword arguments for matplotlib.figure.Figure.savefig
            (http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.savefig).
        fig_alpha : float between 0 and 1, optional
            Transparency of the background of figure but NOT the background of
            the actual plots. 1 means fully opaque, 0 means fully transparent.
        plot_alpha : float between 0 and 1, optional
            Transparency of the background of plots but NOT the background of
            the area surrounding the plots. 1 means fully opaque,
            0 means fully transparent.
        pdf : matplotlib.backends.backend_pdf.PdfPages, optional
            Save the figure to a PDF. Multiple figures from
            multiple multiplotter instances can save to the same PDF.
        """
        current_drive = os.getcwd()
        try:
            os.chdir(save_directory)
        except OSError:
            print("\nSave directory does not exist.\n")
            return -1

        if "dpi" not in save_args:
            save_args["dpi"] = 300

        self._prepare_fig_for_display()

        # set figure background transparency
        self._figure.patch.set_alpha(fig_alpha)
        # set subplot transparency
        for plot_item in self._plot_list:
            plot_item.patch.set_alpha(plot_alpha)

        if pdf is None:
            if "format" not in save_args:
                save_args["format"] = "png"
            if filename is None:
                filename = (self._figure.canvas.get_window_title() + "." +
                            save_args["format"])

            self._figure.savefig(filename,
                                 facecolor=self._figure.get_facecolor(),
                                 **save_args)
        else:
            self._figure.savefig(pdf, format='pdf', **save_args)
        os.chdir(current_drive)

    def display(self, hold=False, display_args=dict()):
        """
        Show the figure.

        Before showing the figure, the plot and plot element spacing will be
        optimized to avoid plots or elements from intersecting each other.

        Parameters
        ----------
        hold : boolean, optional, default = False
            When true, figures will stay open and prevent the program from
            continuing until all figure windows are closed.
        display_args : keyword arguments, optional
            Keyword arguments for matplotlib.figure.Figure.show
            (http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure.show).
        """
        self._prepare_fig_for_display()
        self._figure.show(**display_args)
        if hold:
            plt.show()


    def _target_plots_exist(self,target_plots):
        """
        Private method used to check if the target plot inputs exist.
        """
        if isinstance(target_plots, int):
            target_plots = [target_plots]
        elif target_plots == "all":
            target_plots = self._all_plot_indexes
        elif (type(target_plots) != list and type(target_plots) != tuple and
              target_plots != "all"):
            import pdb;pdb.set_trace()
            print("\ntarget_plots must be an int, list, tuple, or 'all'.\n")
            return 0, target_plots

        for plot_id in target_plots:
            try:
                self._plot_list[plot_id]
            except IndexError:
                print("\n%d is not a valid plot index.\n" % plot_id)
                return 0, target_plots
        return 1, target_plots

    def _prepare_fig_for_display(self):
        """
        Ensure that the spacing of the plots and other elements look "good".
        """
        if self._tighten_layout is True:
            # try to format the plot "nicely"
            # nicely means that all labels and titles are well spaced and do
            # not intersect
            # this is accomplished using tight_layout()
            # but this function does not format figure legends
            # in order to account for this, we will run tight_layout,
            # add some space at the top for the legend, then shift and scale
            # everything down
            try:
                self._figure.tight_layout()
                if self.shrink_top_inches != 0:
                    # add some space to the top of the plot
                    self._figure.set_size_inches(
                        self._figure.get_size_inches() + np.array(
                            [0., self.shrink_top_inches]))

                    # find percentage of the fig height that we need to
                    # shift and scale the plots
                    fig_height = float(self._figure.get_size_inches()[1])
                    shrink_top_ratio = self.shrink_top_inches / fig_height / 1.

                    num_rows = self._grid_shape[0]
                    # height of first plot; assume each plot has same height
                    plot_height = (
                        float(self._plot_list[0].get_position().height) *
                        fig_height)
                    # amount each plot needs to be shrunk
                    reduce_plot_height_inches = (self.shrink_top_inches * 1. /
                                                 num_rows)
                    reduce_plot_height_ratio = (reduce_plot_height_inches /
                                                plot_height)

                    # scale each plot the same
                    self._scale_plots("all", 1., 1. - reduce_plot_height_ratio)

                    # shift the plots different amounts dependong on their row
                    num_plots = len(self._plot_list)
                    num_cols = self._grid_shape[1]
                    # assign a shift amount based on each plots row
                    # don't shift the bottom row at all
                    # shift the top row the most
                    shift_list = [-shrink_top_ratio *
                                  (num_rows-plot_id/num_cols-1) for plot_id
                                  in range(num_plots)]
                    self._shift_plots("all", 0, shift_list)
            except:
                print("\nCannot automatically format subplot layout.\n")

    def set_limits(self, target_plots, x_limits=[], y_limits=[]):
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        set_x_lims = True
        set_y_lims = True
        if x_limits==[] or x_limits==():
            set_x_lims = False
        if y_limits==[] or y_limits==():
            set_y_lims = False

        if set_x_lims and type(x_limits[0]) != list and type(x_limits[0]) != tuple:
            x_limits = [x_limits]*num_target_plots
        if set_y_lims and type(y_limits[0]) != list and type(y_limits[0]) != tuple:
            y_limits = [y_limits]*num_target_plots

        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]

            if y_limits != []:
                plot_item.set_ylim(y_limits[plot_counter])
            if x_limits != []:
                plot_item.set_xlim(x_limits[plot_counter])

    def _scale_plots(self, target_plots, x_scale=1., y_scale=1.):
        """
        Private method used to scale plots.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        # turn x_scale and y_scale into lists if they are not
        if type(x_scale) == int or type(x_scale) == float:
            x_scale = [x_scale]
        if type(y_scale) == int or type(y_scale) == float:
            y_scale = [y_scale]

        len_x_scale = len(x_scale)
        len_y_scale = len(y_scale)
        if len_x_scale != 1 and len_x_scale < num_target_plots:
            print("\n%d scale factors for x-axis were received." % len_x_scale)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one scale factor for every plot")
            print("or only one factor which will be shared by all plots.\n")
            return -1

        if len_y_scale != 1 and len_y_scale < num_target_plots:
            print("\n%d scale factors for y-axis were received." % len_y_scale)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one scale factor for every plot")
            print("or only one factor which will be shared by all plots.\n")
            return -1

        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]

            if len_x_scale == 1:
                scale_x = x_scale[0]
            else:
                scale_x = x_scale[plot_counter]

            if len_y_scale == 1:
                scale_y = y_scale[0]
            else:
                scale_y = y_scale[plot_counter]

            box = plot_item.get_position()
            plot_item.set_position([box.x0, box.y0,
                                    box.width*scale_x, box.height*scale_y])

        self._tighten_layout = False

    def _shift_plots(self, target_plots, x_shift=0., y_shift=0.):
        """
        Private method used to shift plots around.
        """
        # ensure that the target plots are valid
        ret, target_plots = self._target_plots_exist(target_plots)
        if ret is False:
            return -1
        num_target_plots = len(target_plots)

        # turn x_scale and y_scale into lists if they are not
        if type(x_shift) == int or type(x_shift) == float:
            x_shift = [x_shift]
        if type(y_shift) == int or type(y_shift) == float:
            y_shift = [y_shift]

        len_x_shift = len(x_shift)
        len_y_shift = len(y_shift)
        if len_x_shift != 1 and len_x_shift < num_target_plots:
            print("\n%d shift factors for x-axis were received." % len_x_shift)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one shift factor for every plot")
            print("or only one factor which will be shared by all plots.\n")
            return -1

        if len_y_shift != 1 and len_y_shift < num_target_plots:
            print("\n%d shift factors for y-axis were received." % len_y_shift)
            print("%d target plots were received." % num_target_plots)
            print("There either must be one shift factor for every plot")
            print("or only one factor which will be shared by all plots.\n")
            return -1

        for plot_counter, plot_id in enumerate(target_plots):
            plot_item = self._plot_list[plot_id]

            if len_x_shift == 1:
                shift_x = x_shift[0]
            else:
                shift_x = x_shift[plot_counter]

            if len_y_shift == 1:
                shift_y = y_shift[0]
            else:
                shift_y = y_shift[plot_counter]

            box = plot_item.get_position()
            plot_item.set_position([box.x0+box.width*shift_x,
                                    box.y0+box.height*shift_y,
                                    box.width, box.height])

        self._tighten_layout = False

    def add_figure_legend(self, legend_args=dict(), legend_space_inches=0.3,
                          labels=None):
        """
        Add a figure legend.

        Parameters
        ----------
        legend_args : keyword arguments, optional
            Keyword arguments for matplotlib.axes.Axes.legend
            (http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.legend).
        legend_space_inches : float, default: 0.5
            Amount of space, in inches, for the legend at the top of the
            figure.
        labels : string or tuple of strings, optional
            The items in `labels` will be used to populate the figure legend.
            If not provided, the labels set while "adding data" will be used.
        """

        # for now only upper center legends are supported
        legend_args["loc"] = 'upper center'

        # get a list of all "uniquely labeled" lines
        line_label_list = []
        plot_lines = []
        for plot in  self.get_plots("all"):
            for line in plot.lines:
                label = line.get_label()
                if label not in line_label_list:
                    line_label_list.append(label)
                    plot_lines.append(line)

        if labels != None:
            line_label_list = labels

        if "ncol" not in legend_args:
            legend_args["ncol"] = len(line_label_list)

        # create the legend
        self._figure.legend(plot_lines, line_label_list, **legend_args)

        # will need to allocate space for the legend
        # this will be taken care of in _prepare_fig_for_display(),
        # but must set the amount of space to allocate now
        self.shrink_top_inches = legend_space_inches
